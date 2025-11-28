//! Synthetic NER Test Datasets
#![allow(missing_docs)] // Internal evaluation types
//!
//! # Research Context
//!
//! Synthetic data has known limitations (arXiv:2505.16814 "Does Synthetic Data Help NER"):
//!
//! | Issue | Mitigation |
//! |-------|------------|
//! | Entity type skew | Stratified sampling |
//! | Clean annotations | Add noise injection |
//! | Domain gap | Mix with real data |
//! | Label shift | Track via `LabelShift` |
//!
//! # What This Dataset IS Good For
//!
//! - **Unit testing**: Does the code work at all?
//! - **Pattern coverage**: Are regex patterns correct?
//! - **Edge cases**: Unicode, boundaries, special chars
//! - **Fast iteration**: Runs in <1s, no network
//!
//! # What This Dataset IS NOT Good For
//!
//! - **Zero-shot claims**: Label overlap with training ≈ 100%
//! - **Real-world performance**: Synthetic ≠ domain-specific noise
//! - **Model comparison**: Needs WikiGold/CoNLL/WNUT for fair eval
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
    /// Cybersecurity: CVEs, malware, threat actors, attack types
    Cybersecurity,
    /// Multilingual: Non-English text with native scripts
    Multilingual,
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

// Biomedical entity type helpers
fn disease(text: &str, start: usize) -> GoldEntity {
    GoldEntity::new(
        text,
        EntityType::Custom {
            name: "DISEASE".to_string(),
            category: crate::EntityCategory::Misc, // Biomedical entities
        },
        start,
    )
}

fn drug(text: &str, start: usize) -> GoldEntity {
    GoldEntity::new(
        text,
        EntityType::Custom {
            name: "DRUG".to_string(),
            category: crate::EntityCategory::Misc, // Biomedical entities
        },
        start,
    )
}

fn gene(text: &str, start: usize) -> GoldEntity {
    GoldEntity::new(
        text,
        EntityType::Custom {
            name: "GENE".to_string(),
            category: crate::EntityCategory::Misc, // Biomedical entities
        },
        start,
    )
}

fn chemical(text: &str, start: usize) -> GoldEntity {
    GoldEntity::new(
        text,
        EntityType::Custom {
            name: "CHEMICAL".to_string(),
            category: crate::EntityCategory::Misc, // Biomedical entities
        },
        start,
    )
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
        AnnotatedExample {
            text: "omg Taylor Swift just showed up at Arrowhead Stadium in Kansas City!!!!".into(),
            entities: vec![
                entity("Taylor Swift", EntityType::Person, 4),
                entity("Arrowhead Stadium", EntityType::Location, 35),
                entity("Kansas City", EntityType::Location, 56),
            ],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "lol Amazon Prime Day deals r insane this year $50 off everything".into(),
            entities: vec![
                entity("Amazon", EntityType::Organization, 4),
                entity("$50", EntityType::Money, 46),
            ],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "caught the sunrise at Golden Gate Bridge SF is just different ngl".into(),
            entities: vec![
                entity("Golden Gate Bridge", EntityType::Location, 22),
                entity("SF", EntityType::Location, 41),
            ],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Follow me on IG @foodie_nyc or check my site https://foodblog.io".into(),
            entities: vec![entity_url("https://foodblog.io", 45)],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Netflix stock down 20% after earnings miss oof".into(),
            entities: vec![
                entity("Netflix", EntityType::Organization, 0),
                entity("20%", EntityType::Percent, 19),
            ],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "im literally at the Louvre rn and mona lisa kinda mid tbh".into(),
            entities: vec![entity("Louvre", EntityType::Location, 20)],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Hard,
        },
    ]
}

// ============================================================================
// Biomedical Dataset
// ============================================================================

/// Biomedical/healthcare domain dataset
pub fn biomedical_dataset() -> Vec<AnnotatedExample> {
    vec![
        // Easy: Organizations
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
        // Medium: Diseases
        AnnotatedExample {
            text: "The patient was diagnosed with Type 2 diabetes mellitus and hypertension.".into(),
            entities: vec![
                disease("Type 2 diabetes mellitus", 31),
                disease("hypertension", 60),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Metformin is the first-line treatment for diabetes. Lisinopril helps control blood pressure.".into(),
            entities: vec![
                drug("Metformin", 0),
                disease("diabetes", 42),
                drug("Lisinopril", 52),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
        // Hard: Genes and chemicals
        AnnotatedExample {
            text: "Mutations in BRCA1 and BRCA2 genes increase breast cancer risk significantly.".into(),
            entities: vec![
                gene("BRCA1", 13),
                gene("BRCA2", 23),
                disease("breast cancer", 44),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "The p53 tumor suppressor gene regulates cell cycle and apoptosis.".into(),
            entities: vec![
                gene("p53", 4),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Acetylsalicylic acid (aspirin) inhibits COX-1 and COX-2 enzymes.".into(),
            entities: vec![
                chemical("Acetylsalicylic acid", 0),
                drug("aspirin", 22),
                gene("COX-1", 40),
                gene("COX-2", 50),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Hard,
        },
        // Adversarial: Complex medical terminology
        AnnotatedExample {
            text: "The TP53-mutated non-small cell lung carcinoma showed response to pembrolizumab immunotherapy.".into(),
            entities: vec![
                gene("TP53", 4),
                disease("non-small cell lung carcinoma", 17),
                drug("pembrolizumab", 66),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Adversarial,
        },
        AnnotatedExample {
            text: "Rheumatoid arthritis patients receiving tocilizumab showed decreased IL-6 levels.".into(),
            entities: vec![
                disease("Rheumatoid arthritis", 0),
                drug("tocilizumab", 40),
                chemical("IL-6", 69),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Adversarial,
        },
        AnnotatedExample {
            text: "EGFR T790M mutation confers resistance to first-generation EGFR-TKIs like gefitinib.".into(),
            entities: vec![
                gene("EGFR", 0),
                gene("EGFR-TKIs", 59),
                drug("gefitinib", 74),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Adversarial,
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
        AnnotatedExample {
            text: "Judge Ketanji Brown Jackson was confirmed to the Supreme Court on April 7, 2022.".into(),
            entities: vec![
                entity("Ketanji Brown Jackson", EntityType::Person, 6),
                entity("Supreme Court", EntityType::Organization, 49),
                entity("April 7, 2022", EntityType::Date, 66),
            ],
            domain: Domain::Legal,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The FTC filed an antitrust case against Meta Platforms in Washington D.C.".into(),
            entities: vec![
                entity("FTC", EntityType::Organization, 4),
                entity("Meta Platforms", EntityType::Organization, 40),
                entity("Washington D.C.", EntityType::Location, 58),
            ],
            domain: Domain::Legal,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Brown v. Board of Education (1954) ended school segregation in America.".into(),
            entities: vec![entity("America", EntityType::Location, 63)],
            domain: Domain::Legal,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Roe v. Wade was overturned by the Supreme Court on June 24, 2022.".into(),
            entities: vec![
                entity("Supreme Court", EntityType::Organization, 34),
                entity("June 24, 2022", EntityType::Date, 51),
            ],
            domain: Domain::Legal,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Miranda rights derive from Miranda v. Arizona (1966).".into(),
            entities: vec![entity("Arizona", EntityType::Location, 38)],
            domain: Domain::Legal,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "The SEC charged Sam Bankman-Fried with securities fraud totaling $8 billion.".into(),
            entities: vec![
                entity("SEC", EntityType::Organization, 4),
                entity("Sam Bankman-Fried", EntityType::Person, 16),
                entity("$8 billion", EntityType::Money, 65),
            ],
            domain: Domain::Legal,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Contact our firm at legal@lawpartners.com or (212) 555-1234 for a consultation.".into(),
            entities: vec![
                entity_email("legal@lawpartners.com", 20),
                entity_phone("(212) 555-1234", 45),
            ],
            domain: Domain::Legal,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Patent US12345678 was filed by Google LLC on March 15, 2023.".into(),
            entities: vec![
                entity("Google LLC", EntityType::Organization, 31),
                entity("March 15, 2023", EntityType::Date, 45),
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
        AnnotatedExample {
            text: "SpaceX launched Starship from Boca Chica, Texas on April 20, 2023.".into(),
            entities: vec![
                entity("SpaceX", EntityType::Organization, 0),
                entity("Boca Chica", EntityType::Location, 30),
                entity("Texas", EntityType::Location, 42),
                entity("April 20, 2023", EntityType::Date, 51),
            ],
            domain: Domain::Scientific,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Einstein published special relativity while working at the Swiss Patent Office in Bern.".into(),
            entities: vec![
                entity("Einstein", EntityType::Person, 0),
                entity("Swiss Patent Office", EntityType::Organization, 59),
                entity("Bern", EntityType::Location, 82),
            ],
            domain: Domain::Scientific,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "CERN's Large Hadron Collider near Geneva discovered the Higgs boson.".into(),
            entities: vec![
                entity("CERN", EntityType::Organization, 0),
                entity("Geneva", EntityType::Location, 34),
            ],
            domain: Domain::Scientific,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Marie Curie conducted radioactivity research at the University of Paris.".into(),
            entities: vec![
                entity("Marie Curie", EntityType::Person, 0),
                entity("University of Paris", EntityType::Organization, 52),
            ],
            domain: Domain::Scientific,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The Mars Perseverance rover landed in Jezero Crater on February 18, 2021.".into(),
            entities: vec![
                entity("Jezero Crater", EntityType::Location, 38),
                entity("February 18, 2021", EntityType::Date, 55),
            ],
            domain: Domain::Scientific,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "DeepMind's AlphaFold predicted 200 million protein structures.".into(),
            entities: vec![entity("DeepMind", EntityType::Organization, 0)],
            domain: Domain::Scientific,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Prof. Katalin Karikó received the Nobel Prize for mRNA vaccine research.".into(),
            entities: vec![entity("Prof. Katalin Karikó", EntityType::Person, 0)],
            domain: Domain::Scientific,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The Voyager 1 spacecraft, launched in 1977, is now 15 billion miles from Earth.".into(),
            entities: vec![entity("Earth", EntityType::Location, 73)],
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
                entity_email("support@example.com", 8),
                entity_email("sales@company.org", 31),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Easy,
        },
        // URLs
        AnnotatedExample {
            text: "Visit https://docs.example.com or http://api.test.io/v2 for documentation.".into(),
            entities: vec![
                entity_url("https://docs.example.com", 6),
                entity_url("http://api.test.io/v2", 34),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Easy,
        },
        // Phones
        AnnotatedExample {
            text: "Call (555) 123-4567 or +1-800-555-0199 for support.".into(),
            entities: vec![
                entity_phone("(555) 123-4567", 5),
                entity_phone("+1-800-555-0199", 23),
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
                entity_email("billing@corp.com", 66),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Webinar on Jan 20 at 2pm EST. Register at https://events.co/webinar. Fee: $99.".into(),
            entities: vec![
                entity("Jan 20", EntityType::Date, 11),
                entity("2pm", EntityType::Date, 21),
                entity_url("https://events.co/webinar", 42),
                entity("$99", EntityType::Money, 74),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Medium,
        },
    ]
}

/// Helper for creating email entities
fn entity_email(text: &str, start: usize) -> GoldEntity {
    GoldEntity::new(text, EntityType::Email, start)
}

/// Helper for creating URL entities
fn entity_url(text: &str, start: usize) -> GoldEntity {
    GoldEntity::new(text, EntityType::Url, start)
}

/// Helper for creating phone entities
fn entity_phone(text: &str, start: usize) -> GoldEntity {
    GoldEntity::new(text, EntityType::Phone, start)
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
        AnnotatedExample {
            text: "Rafael Nadal defeated Roger Federer at Roland Garros in Paris.".into(),
            entities: vec![
                entity("Rafael Nadal", EntityType::Person, 0),
                entity("Roger Federer", EntityType::Person, 22),
                entity("Roland Garros", EntityType::Location, 39),
                entity("Paris", EntityType::Location, 56),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Golden State Warriors beat the Miami Heat 110-95 on December 3, 2023.".into(),
            entities: vec![
                entity("Golden State Warriors", EntityType::Organization, 4),
                entity("Miami Heat", EntityType::Organization, 35),
                entity("December 3, 2023", EntityType::Date, 56),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Usain Bolt ran 9.58 seconds at the Berlin World Championships.".into(),
            entities: vec![
                entity("Usain Bolt", EntityType::Person, 0),
                entity("Berlin", EntityType::Location, 35),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The NFL fined Tom Brady $10,000 for equipment violations on January 5.".into(),
            entities: vec![
                entity("NFL", EntityType::Organization, 4),
                entity("Tom Brady", EntityType::Person, 14),
                entity("$10,000", EntityType::Money, 24),
                entity("January 5", EntityType::Date, 60),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Simone Biles won gold for the USA at the Tokyo Olympics.".into(),
            entities: vec![
                entity("Simone Biles", EntityType::Person, 0),
                entity("USA", EntityType::Organization, 30),
                entity("Tokyo", EntityType::Location, 41),
            ],
            domain: Domain::Sports,
            difficulty: Difficulty::Easy,
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
        AnnotatedExample {
            text: "Steven Spielberg directed Jurassic Park for Universal Pictures.".into(),
            entities: vec![
                entity("Steven Spielberg", EntityType::Person, 0),
                entity("Universal Pictures", EntityType::Organization, 44),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "BTS performed at SoFi Stadium in Los Angeles on November 27, 2021.".into(),
            entities: vec![
                entity("BTS", EntityType::Organization, 0),
                entity("SoFi Stadium", EntityType::Location, 17),
                entity("Los Angeles", EntityType::Location, 33),
                entity("November 27, 2021", EntityType::Date, 48),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Warner Bros released Dune directed by Denis Villeneuve.".into(),
            entities: vec![
                entity("Warner Bros", EntityType::Organization, 0),
                entity("Denis Villeneuve", EntityType::Person, 38),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The Oscars nominated Cillian Murphy for his role in Oppenheimer.".into(),
            entities: vec![
                entity("Cillian Murphy", EntityType::Person, 21),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Spotify paid $100 million for Joe Rogan's podcast deal in Austin.".into(),
            entities: vec![
                entity("Spotify", EntityType::Organization, 0),
                entity("$100 million", EntityType::Money, 13),
                entity("Joe Rogan", EntityType::Person, 30),
                entity("Austin", EntityType::Location, 58),
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
                entity_email("help@store.com", 28),
                entity_phone("(800) 555-1234", 46),
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
                entity_url("https://travel.example.com", 8),
                entity_phone("+1-888-555-0100", 43),
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
        AnnotatedExample {
            text: "Tornado warning issued for Oklahoma City until 6pm on May 3, 2024.".into(),
            entities: vec![
                entity("Oklahoma City", EntityType::Location, 27),
                entity("6pm", EntityType::Date, 47),
                entity("May 3, 2024", EntityType::Date, 54),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "AccuWeather predicts 6 inches of snow in Denver this weekend.".into(),
            entities: vec![
                entity("AccuWeather", EntityType::Organization, 0),
                entity("Denver", EntityType::Location, 41),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The National Weather Service warns of flooding in Houston, Texas.".into(),
            entities: vec![
                entity("National Weather Service", EntityType::Organization, 4),
                entity("Houston", EntityType::Location, 50),
                entity("Texas", EntityType::Location, 59),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Typhoon Haiyan devastated the Philippines in November 2013.".into(),
            entities: vec![
                entity("Philippines", EntityType::Location, 30),
                entity("November 2013", EntityType::Date, 45),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Check forecasts at https://weather.gov or call (800) 555-1234.".into(),
            entities: vec![
                entity_url("https://weather.gov", 19),
                entity_phone("(800) 555-1234", 47),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Record high of 130°F in Death Valley, California on July 10, 2021.".into(),
            entities: vec![
                entity("Death Valley", EntityType::Location, 24),
                entity("California", EntityType::Location, 38),
                entity("July 10, 2021", EntityType::Date, 52),
            ],
            domain: Domain::Weather,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "FEMA deployed to Louisiana after Hurricane Ida on August 29, 2021.".into(),
            entities: vec![
                entity("FEMA", EntityType::Organization, 0),
                entity("Louisiana", EntityType::Location, 17),
                entity("August 29, 2021", EntityType::Date, 50),
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
                entity_email("admissions@university.edu", 45),
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
        AnnotatedExample {
            text: "Winston Churchill gave the Iron Curtain speech in Fulton, Missouri on March 5, 1946.".into(),
            entities: vec![
                entity("Winston Churchill", EntityType::Person, 0),
                entity("Fulton", EntityType::Location, 50),
                entity("Missouri", EntityType::Location, 58),
                entity("March 5, 1946", EntityType::Date, 70),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Cleopatra ruled Egypt from Alexandria until 30 BC.".into(),
            entities: vec![
                entity("Cleopatra", EntityType::Person, 0),
                entity("Egypt", EntityType::Location, 16),
                entity("Alexandria", EntityType::Location, 27),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Martin Luther King Jr. delivered the I Have a Dream speech in Washington D.C. on August 28, 1963.".into(),
            entities: vec![
                entity("Martin Luther King Jr.", EntityType::Person, 0),
                entity("Washington D.C.", EntityType::Location, 62),
                entity("August 28, 1963", EntityType::Date, 81),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Berlin Wall fell on November 9, 1989.".into(),
            entities: vec![
                entity("Berlin", EntityType::Location, 4),
                entity("November 9, 1989", EntityType::Date, 24),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Mahatma Gandhi led India's independence movement against the British Empire.".into(),
            entities: vec![
                entity("Mahatma Gandhi", EntityType::Person, 0),
                entity("India", EntityType::Location, 19),
                entity("British Empire", EntityType::Organization, 61),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Magna Carta was signed at Runnymede, England on June 15, 1215.".into(),
            entities: vec![
                entity("Runnymede", EntityType::Location, 30),
                entity("England", EntityType::Location, 41),
                entity("June 15, 1215", EntityType::Date, 52),
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
                entity_url("https://food.delivery", 16),
                entity_phone("(555) 123-4567", 46),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Chef Gordon Ramsay opened a new restaurant in Las Vegas.".into(),
            entities: vec![
                entity("Gordon Ramsay", EntityType::Person, 5),
                entity("Las Vegas", EntityType::Location, 46),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Starbucks CEO Laxman Narasimhan announced $3 billion in buybacks.".into(),
            entities: vec![
                entity("Starbucks", EntityType::Organization, 0),
                entity("Laxman Narasimhan", EntityType::Person, 14),
                entity("$3 billion", EntityType::Money, 42),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Chipotle Mexican Grill opened 100 new locations in California.".into(),
            entities: vec![
                entity("Chipotle Mexican Grill", EntityType::Organization, 0),
                entity("California", EntityType::Location, 51),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "DoorDash charges 15% commission to restaurants in New York City.".into(),
            entities: vec![
                entity("DoorDash", EntityType::Organization, 0),
                entity("15%", EntityType::Percent, 17),
                entity("New York City", EntityType::Location, 50),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Sweetgreen raised $200 million at a $3.2 billion valuation.".into(),
            entities: vec![
                entity("Sweetgreen", EntityType::Organization, 0),
                entity("$200 million", EntityType::Money, 18),
                entity("$3.2 billion", EntityType::Money, 36),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Contact reservations@finedining.com for bookings on January 15, 2025.".into(),
            entities: vec![
                entity_email("reservations@finedining.com", 8),
                entity("January 15, 2025", EntityType::Date, 52),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Yum! Brands owns KFC, Pizza Hut, and Taco Bell worldwide.".into(),
            entities: vec![
                entity("Yum! Brands", EntityType::Organization, 0),
                entity("KFC", EntityType::Organization, 17),
                entity("Pizza Hut", EntityType::Organization, 22),
                entity("Taco Bell", EntityType::Organization, 37),
            ],
            domain: Domain::Food,
            difficulty: Difficulty::Hard,
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
                entity_email("agent@realty.com", 59),
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
        AnnotatedExample {
            text: "Redfin CEO Glenn Kelman announced 20% workforce reduction.".into(),
            entities: vec![
                entity("Redfin", EntityType::Organization, 0),
                entity("Glenn Kelman", EntityType::Person, 11),
                entity("20%", EntityType::Percent, 34),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Blackstone acquired $6 billion in apartments in Miami and Atlanta.".into(),
            entities: vec![
                entity("Blackstone", EntityType::Organization, 0),
                entity("$6 billion", EntityType::Money, 20),
                entity("Miami", EntityType::Location, 48),
                entity("Atlanta", EntityType::Location, 58),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Call (310) 555-7890 for property tours in Los Angeles.".into(),
            entities: vec![
                entity_phone("(310) 555-7890", 5),
                entity("Los Angeles", EntityType::Location, 42),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "CBRE reports commercial vacancy rates at 18% in Manhattan.".into(),
            entities: vec![
                entity("CBRE", EntityType::Organization, 0),
                entity("18%", EntityType::Percent, 41),
                entity("Manhattan", EntityType::Location, 48),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Visit https://homes.com for listings or email info@homes.com.".into(),
            entities: vec![
                entity_url("https://homes.com", 6),
                entity_email("info@homes.com", 46),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "JLL predicts 5% rent growth in Seattle through 2025.".into(),
            entities: vec![
                entity("JLL", EntityType::Organization, 0),
                entity("5%", EntityType::Percent, 13),
                entity("Seattle", EntityType::Location, 31),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "CoStar acquired RentPath for $588 million in February 2021.".into(),
            entities: vec![
                entity("CoStar", EntityType::Organization, 0),
                entity("RentPath", EntityType::Organization, 16),
                entity("$588 million", EntityType::Money, 29),
                entity("February 2021", EntityType::Date, 45),
            ],
            domain: Domain::RealEstate,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Cybersecurity Dataset
// ============================================================================

/// Cybersecurity domain: CVEs, malware, threat actors, attack types
pub fn cybersecurity_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "CVE-2024-3094 affects XZ Utils versions 5.6.0 and 5.6.1.".into(),
            entities: vec![
                entity("CVE-2024-3094", EntityType::custom("CVE", crate::EntityCategory::Misc), 0),
                entity("XZ Utils", EntityType::custom("SOFTWARE", crate::EntityCategory::Misc), 22),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Lazarus Group targeted Sony Pictures in the 2014 attack.".into(),
            entities: vec![
                entity("Lazarus Group", EntityType::Organization, 4),
                entity("Sony Pictures", EntityType::Organization, 27),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Microsoft patched CVE-2023-23397 in Outlook on March 14, 2023.".into(),
            entities: vec![
                entity("Microsoft", EntityType::Organization, 0),
                entity("CVE-2023-23397", EntityType::custom("CVE", crate::EntityCategory::Misc), 18),
                entity("Outlook", EntityType::custom("SOFTWARE", crate::EntityCategory::Misc), 36),
                entity("March 14, 2023", EntityType::Date, 47),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "WannaCry ransomware exploited EternalBlue to infect 200,000 systems.".into(),
            entities: vec![
                entity("WannaCry", EntityType::custom("MALWARE", crate::EntityCategory::Misc), 0),
                entity("EternalBlue", EntityType::custom("EXPLOIT", crate::EntityCategory::Misc), 30),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "CISA issued an advisory for Log4Shell (CVE-2021-44228) on December 10.".into(),
            entities: vec![
                entity("CISA", EntityType::Organization, 0),
                entity("Log4Shell", EntityType::custom("VULNERABILITY", crate::EntityCategory::Misc), 28),
                entity("CVE-2021-44228", EntityType::custom("CVE", crate::EntityCategory::Misc), 39),
                entity("December 10", EntityType::Date, 58),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "APT29 (Cozy Bear) breached SolarWinds using SUNBURST malware.".into(),
            entities: vec![
                entity("APT29", EntityType::custom("THREAT_ACTOR", crate::EntityCategory::Organization), 0),
                entity("Cozy Bear", EntityType::custom("THREAT_ACTOR", crate::EntityCategory::Organization), 7),
                entity("SolarWinds", EntityType::Organization, 27),
                entity("SUNBURST", EntityType::custom("MALWARE", crate::EntityCategory::Misc), 44),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Cisco Talos discovered the BlackCat/ALPHV ransomware variant.".into(),
            entities: vec![
                entity("Cisco Talos", EntityType::Organization, 0),
                entity("BlackCat", EntityType::custom("MALWARE", crate::EntityCategory::Misc), 27),
                entity("ALPHV", EntityType::custom("MALWARE", crate::EntityCategory::Misc), 36),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The FBI attributed the Colonial Pipeline attack to DarkSide.".into(),
            entities: vec![
                entity("FBI", EntityType::Organization, 4),
                entity("Colonial Pipeline", EntityType::Organization, 23),
                entity("DarkSide", EntityType::custom("THREAT_ACTOR", crate::EntityCategory::Organization), 51),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "CrowdStrike reported Fancy Bear targeting NATO in 2024.".into(),
            entities: vec![
                entity("CrowdStrike", EntityType::Organization, 0),
                entity("Fancy Bear", EntityType::custom("THREAT_ACTOR", crate::EntityCategory::Organization), 21),
                entity("NATO", EntityType::Organization, 42),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Mandiant tracked UNC2452 behind the SolarWinds compromise.".into(),
            entities: vec![
                entity("Mandiant", EntityType::Organization, 0),
                entity("UNC2452", EntityType::custom("THREAT_ACTOR", crate::EntityCategory::Organization), 17),
                entity("SolarWinds", EntityType::Organization, 36),
            ],
            domain: Domain::Cybersecurity,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Multilingual Dataset (Native Scripts)
// ============================================================================

/// Multilingual examples with native scripts (Chinese, Japanese, Russian, Arabic, Korean)
pub fn multilingual_dataset() -> Vec<AnnotatedExample> {
    vec![
        // === Chinese (Simplified) ===
        AnnotatedExample {
            text: "华为公司在深圳成立于1987年。".into(),
            entities: vec![
                entity("华为公司", EntityType::Organization, 0),
                entity("深圳", EntityType::Location, 5),
                entity("1987年", EntityType::Date, 10),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "习近平主席访问了北京大学。".into(),
            entities: vec![
                entity("习近平", EntityType::Person, 0),
                entity("北京大学", EntityType::Organization, 8),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "腾讯以$400亿收购了上海的一家公司。".into(),
            entities: vec![
                entity("腾讯", EntityType::Organization, 0),
                entity("$400亿", EntityType::Money, 3),
                entity("上海", EntityType::Location, 11),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Hard,
        },

        // === Japanese ===
        AnnotatedExample {
            text: "トヨタ自動車は2024年3月に名古屋で発表した。".into(),
            entities: vec![
                entity("トヨタ自動車", EntityType::Organization, 0),
                entity("2024年3月", EntityType::Date, 7),
                entity("名古屋", EntityType::Location, 15),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "山田太郎氏がソニーの新CEOに就任した。".into(),
            entities: vec![
                entity("山田太郎", EntityType::Person, 0),
                entity("ソニー", EntityType::Organization, 6),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "東京オリンピックは2021年に開催された。".into(),
            entities: vec![
                entity("東京", EntityType::Location, 0),
                entity("2021年", EntityType::Date, 9),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Easy,
        },

        // === Russian (Cyrillic) ===
        AnnotatedExample {
            text: "Газпром объявил о сделке на $50 млрд в Москве.".into(),
            entities: vec![
                entity("Газпром", EntityType::Organization, 0),
                entity("$50 млрд", EntityType::Money, 28),
                entity("Москве", EntityType::Location, 39),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Владимир Путин встретился с Сергеем Лавровым в Кремле.".into(),
            entities: vec![
                entity("Владимир Путин", EntityType::Person, 0),
                entity("Сергеем Лавровым", EntityType::Person, 28),
                entity("Кремле", EntityType::Location, 47),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Яндекс открыл офис в Санкт-Петербурге 15 марта 2024.".into(),
            entities: vec![
                entity("Яндекс", EntityType::Organization, 0),
                entity("Санкт-Петербурге", EntityType::Location, 21),
                entity("15 марта 2024", EntityType::Date, 38),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },

        // === Arabic ===
        AnnotatedExample {
            text: "أعلنت شركة أرامكو السعودية عن أرباح بقيمة $100 مليار.".into(),
            entities: vec![
                entity("أرامكو السعودية", EntityType::Organization, 11),
                entity("$100 مليار", EntityType::Money, 42),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "زار محمد بن سلمان القاهرة في يناير 2024.".into(),
            entities: vec![
                entity("محمد بن سلمان", EntityType::Person, 4),
                entity("القاهرة", EntityType::Location, 18),
                entity("يناير 2024", EntityType::Date, 29),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },

        // === Korean (Hangul) ===
        AnnotatedExample {
            text: "삼성전자가 서울에서 신제품을 발표했다.".into(),
            entities: vec![
                entity("삼성전자", EntityType::Organization, 0),
                entity("서울", EntityType::Location, 6),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "윤석열 대통령이 2024년 3월 도쿄를 방문했다.".into(),
            entities: vec![
                entity("윤석열", EntityType::Person, 0),
                entity("2024년 3월", EntityType::Date, 9),
                entity("도쿄", EntityType::Location, 18),
            ],
            domain: Domain::Multilingual,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "현대자동차는 $10억 투자를 발표했다.".into(),
            entities: vec![
                entity("현대자동차", EntityType::Organization, 0),
                entity("$10억", EntityType::Money, 7),
            ],
            domain: Domain::Multilingual,
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
                entity_email("john@email.com", 26),
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
        AnnotatedExample {
            text: "Schedule a meeting with Sarah at Google for Friday.".into(),
            entities: vec![
                entity("Sarah", EntityType::Person, 24),
                entity("Google", EntityType::Organization, 33),
                entity("Friday", EntityType::Date, 44),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Order me a pizza from Domino's to 123 Main Street.".into(),
            entities: vec![
                entity("Domino's", EntityType::Organization, 22),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Find restaurants near Central Park that cost under $30.".into(),
            entities: vec![
                entity("Central Park", EntityType::Location, 22),
                entity("$30", EntityType::Money, 51),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Call Mom at her Boston number after 5pm on Saturday.".into(),
            entities: vec![
                entity("Mom", EntityType::Person, 5),
                entity("Boston", EntityType::Location, 16),
                entity("5pm", EntityType::Date, 36),
                entity("Saturday", EntityType::Date, 43),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "How much did Amazon stock rise today?".into(),
            entities: vec![
                entity("Amazon", EntityType::Organization, 13),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Book a table at The French Laundry for 7pm on Valentine's Day.".into(),
            entities: vec![
                entity("The French Laundry", EntityType::Organization, 16),
                entity("7pm", EntityType::Date, 39),
                entity("Valentine's Day", EntityType::Date, 46),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Extended High-Quality Dataset
// ============================================================================

/// Extended dataset with high-quality annotations for comprehensive testing.
///
/// Focuses on:
/// - Unicode/multilingual text
/// - Complex structured entities
/// - Nested and overlapping entity contexts
/// - Edge cases for evaluation
pub fn extended_quality_dataset() -> Vec<AnnotatedExample> {
    vec![
        // === Unicode/Multilingual ===
        AnnotatedExample {
            text: "François Hollande met Angela Merkel in Paris on 15 janvier 2024.".into(),
            entities: vec![
                entity("François Hollande", EntityType::Person, 0),
                entity("Angela Merkel", EntityType::Person, 22),
                entity("Paris", EntityType::Location, 39),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "株式会社トヨタ announced ¥500億 investment in 東京.".into(),
            entities: vec![
                entity("株式会社トヨタ", EntityType::Organization, 0),
                entity("¥500億", EntityType::Money, 18),
                entity("東京", EntityType::Location, 38),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Россия и Китай подписали соглашение в Москве.".into(),
            entities: vec![
                entity("Россия", EntityType::Location, 0),
                entity("Китай", EntityType::Location, 9),
                entity("Москве", EntityType::Location, 38),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Hard,
        },
        
        // === Complex Structured Entities ===
        AnnotatedExample {
            text: "Invoice #2024-001: $15,432.50 due by March 31, 2024.".into(),
            entities: vec![
                entity("$15,432.50", EntityType::Money, 19),
                entity("March 31, 2024", EntityType::Date, 37),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Contact support@acme-corp.io or call +1 (555) 123-4567.".into(),
            entities: vec![
                entity("support@acme-corp.io", EntityType::Email, 8),
                entity("+1 (555) 123-4567", EntityType::Phone, 37),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Meeting at 2:30 PM EST on 2024-01-15 at 123 Main St, Suite 400.".into(),
            entities: vec![
                entity("2:30 PM", EntityType::Time, 11),
                entity("2024-01-15", EntityType::Date, 26),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Price: €1.299,99 (incl. 19% VAT) - Delivery by 15/03/2024.".into(),
            entities: vec![
                entity("€1.299,99", EntityType::Money, 7),
                entity("19%", EntityType::Percent, 24),
                entity("15/03/2024", EntityType::Date, 47),
            ],
            domain: Domain::Ecommerce,
            difficulty: Difficulty::Medium,
        },
        
        // === Nested Entity Contexts ===
        AnnotatedExample {
            text: "The Bank of America CEO spoke at the University of California, Berkeley.".into(),
            entities: vec![
                entity("Bank of America", EntityType::Organization, 4),
                entity("University of California, Berkeley", EntityType::Organization, 37),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Dr. Sarah Johnson, MD, PhD, from Johns Hopkins Hospital in Baltimore.".into(),
            entities: vec![
                entity("Dr. Sarah Johnson", EntityType::Person, 0),
                entity("Johns Hopkins Hospital", EntityType::Organization, 33),
                entity("Baltimore", EntityType::Location, 59),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Hard,
        },
        
        // === Tricky Entity Boundaries ===
        AnnotatedExample {
            text: "Apple's $3B acquisition vs. Google's $2.1B offer for DeepMind.".into(),
            entities: vec![
                entity("Apple", EntityType::Organization, 0),
                entity("$3B", EntityType::Money, 8),
                entity("Google", EntityType::Organization, 28),
                entity("$2.1B", EntityType::Money, 37),
                entity("DeepMind", EntityType::Organization, 53),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "From NYC to LA: 2,451 miles, ~$500 by flight, arriving Jan. 15th.".into(),
            entities: vec![
                entity("NYC", EntityType::Location, 5),
                entity("LA", EntityType::Location, 12),
                entity("$500", EntityType::Money, 30),
                entity("Jan. 15th", EntityType::Date, 55),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Medium,
        },
        
        // === Possessives and Contractions ===
        AnnotatedExample {
            text: "Microsoft's Satya Nadella and Amazon's Andy Jassy discussed AI.".into(),
            entities: vec![
                entity("Microsoft", EntityType::Organization, 0),
                entity("Satya Nadella", EntityType::Person, 12),
                entity("Amazon", EntityType::Organization, 30),
                entity("Andy Jassy", EntityType::Person, 39),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        
        // === Abbreviations and Acronyms ===
        AnnotatedExample {
            text: "The EU and US signed a $50M agreement with NATO in Brussels.".into(),
            entities: vec![
                entity("EU", EntityType::Organization, 4),
                entity("US", EntityType::Location, 11),
                entity("$50M", EntityType::Money, 23),
                entity("NATO", EntityType::Organization, 43),
                entity("Brussels", EntityType::Location, 51),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
        
        // === Edge Cases ===
        AnnotatedExample {
            text: "  Whitespace   around   entities   like   Apple   matters.  ".into(),
            entities: vec![
                entity("Apple", EntityType::Organization, 42),
            ],
            domain: Domain::Technical,
            difficulty: Difficulty::Adversarial,
        },
        AnnotatedExample {
            text: "ALL CAPS: MICROSOFT ANNOUNCED $100M FOR SEATTLE EXPANSION.".into(),
            entities: vec![
                entity("MICROSOFT", EntityType::Organization, 10),
                entity("$100M", EntityType::Money, 30),
                entity("SEATTLE", EntityType::Location, 40),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        AnnotatedExample {
            text: "lowercase: apple ceo tim cook visited new york city last january.".into(),
            entities: vec![
                // Note: All lowercase - difficult case, annotators differ
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        
        // === Multiple Same-Type Entities ===
        AnnotatedExample {
            text: "Meeting with John, Mary, Bob, and Alice at 3pm to discuss the $1M budget.".into(),
            entities: vec![
                entity("John", EntityType::Person, 13),
                entity("Mary", EntityType::Person, 19),
                entity("Bob", EntityType::Person, 25),
                entity("Alice", EntityType::Person, 34),
                entity("3pm", EntityType::Time, 43),
                entity("$1M", EntityType::Money, 62),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Medium,
        },
        
        // === Long Entity Spans ===
        AnnotatedExample {
            text: "The Massachusetts Institute of Technology and Carnegie Mellon University collaborated.".into(),
            entities: vec![
                entity("Massachusetts Institute of Technology", EntityType::Organization, 4),
                entity("Carnegie Mellon University", EntityType::Organization, 46),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        
        // === Numbers in Context ===
        AnnotatedExample {
            text: "Version 3.14.159 costs $99.99 (50% off from $199.99) until Dec 31.".into(),
            entities: vec![
                entity("$99.99", EntityType::Money, 23),
                entity("50%", EntityType::Percent, 31),
                entity("$199.99", EntityType::Money, 44),
                entity("Dec 31", EntityType::Date, 59),
            ],
            domain: Domain::Ecommerce,
            difficulty: Difficulty::Medium,
        },
        
        // === Quotes and Attribution ===
        AnnotatedExample {
            text: "\"We're excited,\" said Elon Musk from Tesla's Austin headquarters.".into(),
            entities: vec![
                entity("Elon Musk", EntityType::Person, 22),
                entity("Tesla", EntityType::Organization, 37),
                entity("Austin", EntityType::Location, 45),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        
        // === Medical/Technical ===
        AnnotatedExample {
            text: "Patient received 500mg aspirin at 08:30 on 2024-01-15 at Mayo Clinic.".into(),
            entities: vec![
                entity("08:30", EntityType::Time, 34),
                entity("2024-01-15", EntityType::Date, 43),
                entity("Mayo Clinic", EntityType::Organization, 57),
            ],
            domain: Domain::Biomedical,
            difficulty: Difficulty::Medium,
        },
    ]
}

// ============================================================================
// Globally Diverse Dataset (Bias Testing)
// ============================================================================

/// Globally diverse dataset for demographic bias testing.
///
/// Includes names and entities from multiple cultural/ethnic backgrounds
/// to test for systematic bias in NER systems.
///
/// Based on research showing NER models perform worse on:
/// - Non-Western names (Mishra et al. 2020)
/// - Non-Latin scripts (Loessberg-Zahl 2024)
/// - Rare or uncommon name patterns
pub fn globally_diverse_dataset() -> Vec<AnnotatedExample> {
    vec![
        // === African Names ===
        AnnotatedExample {
            text: "Chidi Okonkwo is the CEO of Lagos Tech Solutions in Nigeria.".into(),
            entities: vec![
                entity("Chidi Okonkwo", EntityType::Person, 0),
                entity("Lagos Tech Solutions", EntityType::Organization, 28),
                entity("Nigeria", EntityType::Location, 52),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Amara Adebayo met Kwame Mensah in Accra to discuss the partnership.".into(),
            entities: vec![
                entity("Amara Adebayo", EntityType::Person, 0),
                entity("Kwame Mensah", EntityType::Person, 18),
                entity("Accra", EntityType::Location, 34),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Oluwaseun Afolabi works at the University of Nairobi in Kenya.".into(),
            entities: vec![
                entity("Oluwaseun Afolabi", EntityType::Person, 0),
                entity("University of Nairobi", EntityType::Organization, 31),
                entity("Kenya", EntityType::Location, 56),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },

        // === South Asian Names ===
        AnnotatedExample {
            text: "Dr. Priya Sharma presented research at IIT Delhi on February 15, 2024.".into(),
            entities: vec![
                entity("Dr. Priya Sharma", EntityType::Person, 0),
                entity("IIT Delhi", EntityType::Organization, 39),
                entity("February 15, 2024", EntityType::Date, 52),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Raj Patel and Arjun Singh founded Mumbai AI Labs with $5 million.".into(),
            entities: vec![
                entity("Raj Patel", EntityType::Person, 0),
                entity("Arjun Singh", EntityType::Person, 14),
                entity("Mumbai AI Labs", EntityType::Organization, 34),
                entity("$5 million", EntityType::Money, 54),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Aisha Khan from Karachi won the award at the Dhaka conference.".into(),
            entities: vec![
                entity("Aisha Khan", EntityType::Person, 0),
                entity("Karachi", EntityType::Location, 16),
                entity("Dhaka", EntityType::Location, 45),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },

        // === East Asian Names (Romanized) ===
        AnnotatedExample {
            text: "Wei Wang and Li Zhang lead Tsinghua University's AI research team.".into(),
            entities: vec![
                entity("Wei Wang", EntityType::Person, 0),
                entity("Li Zhang", EntityType::Person, 13),
                entity("Tsinghua University", EntityType::Organization, 27),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Yuki Tanaka met Min-jun Kim in Seoul to discuss trade with Tokyo.".into(),
            entities: vec![
                entity("Yuki Tanaka", EntityType::Person, 0),
                entity("Min-jun Kim", EntityType::Person, 16),
                entity("Seoul", EntityType::Location, 31),
                entity("Tokyo", EntityType::Location, 59),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Seo-yeon Park is the director of Samsung's Beijing office.".into(),
            entities: vec![
                entity("Seo-yeon Park", EntityType::Person, 0),
                entity("Samsung", EntityType::Organization, 33),
                entity("Beijing", EntityType::Location, 43),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },

        // === Middle Eastern Names ===
        AnnotatedExample {
            text: "Ahmed Hassan founded Dubai Innovations with backing from Abu Dhabi.".into(),
            entities: vec![
                entity("Ahmed Hassan", EntityType::Person, 0),
                entity("Dubai Innovations", EntityType::Organization, 21),
                entity("Abu Dhabi", EntityType::Location, 57),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Fatima Ali and Mohammed Ibrahim met in Tehran on January 10, 2024.".into(),
            entities: vec![
                entity("Fatima Ali", EntityType::Person, 0),
                entity("Mohammed Ibrahim", EntityType::Person, 15),
                entity("Tehran", EntityType::Location, 39),
                entity("January 10, 2024", EntityType::Date, 49),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },

        // === Hispanic/Latino Names ===
        AnnotatedExample {
            text: "José García and María Rodriguez lead UNAM's research in Mexico City.".into(),
            entities: vec![
                entity("José García", EntityType::Person, 0),
                entity("María Rodriguez", EntityType::Person, 16),
                entity("UNAM", EntityType::Organization, 37),
                entity("Mexico City", EntityType::Location, 56),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Carlos Hernández announced $2 billion investment in São Paulo.".into(),
            entities: vec![
                entity("Carlos Hernández", EntityType::Person, 0),
                entity("$2 billion", EntityType::Money, 27),
                entity("São Paulo", EntityType::Location, 52),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Guadalupe Sánchez from Buenos Aires visited Lima on March 5, 2024.".into(),
            entities: vec![
                entity("Guadalupe Sánchez", EntityType::Person, 0),
                entity("Buenos Aires", EntityType::Location, 23),
                entity("Lima", EntityType::Location, 44),
                entity("March 5, 2024", EntityType::Date, 52),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Medium,
        },

        // === African-American Names ===
        AnnotatedExample {
            text: "DeShawn Jackson and Latoya Williams founded Atlanta Tech Hub.".into(),
            entities: vec![
                entity("DeShawn Jackson", EntityType::Person, 0),
                entity("Latoya Williams", EntityType::Person, 20),
                entity("Atlanta Tech Hub", EntityType::Organization, 44),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Jamal Robinson received $3 million funding from Howard University.".into(),
            entities: vec![
                entity("Jamal Robinson", EntityType::Person, 0),
                entity("$3 million", EntityType::Money, 24),
                entity("Howard University", EntityType::Organization, 48),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Aaliyah Washington spoke at Spelman College in Atlanta on May 15.".into(),
            entities: vec![
                entity("Aaliyah Washington", EntityType::Person, 0),
                entity("Spelman College", EntityType::Organization, 28),
                entity("Atlanta", EntityType::Location, 47),
                entity("May 15", EntityType::Date, 58),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },

        // === Eastern European Names ===
        AnnotatedExample {
            text: "Ivan Petrov met Olga Ivanova in Moscow at the Kremlin.".into(),
            entities: vec![
                entity("Ivan Petrov", EntityType::Person, 0),
                entity("Olga Ivanova", EntityType::Person, 16),
                entity("Moscow", EntityType::Location, 32),
                entity("Kremlin", EntityType::Location, 46),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Katarzyna Kowalski from Warsaw visited Prague on April 20, 2024.".into(),
            entities: vec![
                entity("Katarzyna Kowalski", EntityType::Person, 0),
                entity("Warsaw", EntityType::Location, 24),
                entity("Prague", EntityType::Location, 39),
                entity("April 20, 2024", EntityType::Date, 49),
            ],
            domain: Domain::Travel,
            difficulty: Difficulty::Medium,
        },

        // === Mixed/Intersectional Examples ===
        AnnotatedExample {
            text: "Priya Sharma from Mumbai met Wei Wang from Beijing at MIT.".into(),
            entities: vec![
                entity("Priya Sharma", EntityType::Person, 0),
                entity("Mumbai", EntityType::Location, 18),
                entity("Wei Wang", EntityType::Person, 29),
                entity("Beijing", EntityType::Location, 43),
                entity("MIT", EntityType::Organization, 54),
            ],
            domain: Domain::Academic,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Ahmed Hassan and María García discussed $50M partnership in Geneva.".into(),
            entities: vec![
                entity("Ahmed Hassan", EntityType::Person, 0),
                entity("María García", EntityType::Person, 17),
                entity("$50M", EntityType::Money, 40),
                entity("Geneva", EntityType::Location, 60),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Chidi Okonkwo, Yuki Tanaka, and José García presented at the UN in New York.".into(),
            entities: vec![
                entity("Chidi Okonkwo", EntityType::Person, 0),
                entity("Yuki Tanaka", EntityType::Person, 15),
                entity("José García", EntityType::Person, 32),
                entity("UN", EntityType::Organization, 61),
                entity("New York", EntityType::Location, 67),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Hard,
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
    all.extend(cybersecurity_dataset());
    all.extend(multilingual_dataset());
    all.extend(conversational_dataset());
    all.extend(extended_quality_dataset());
    all.extend(globally_diverse_dataset());
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
            let text_chars: Vec<char> = example.text.chars().collect();
            
            for entity in &example.entities {
                // Check bounds
                assert!(
                    entity.end <= text_chars.len(),
                    "Entity '{}' end {} exceeds char count {} in: {}",
                    entity.text,
                    entity.end,
                    text_chars.len(),
                    example.text
                );
                
                // Extract actual text at offset and compare
                let actual_text: String = text_chars[entity.start..entity.end].iter().collect();
                assert_eq!(
                    actual_text, entity.text,
                    "Entity text mismatch at [{}, {}): expected '{}', found '{}' in: {}",
                    entity.start, entity.end, entity.text, actual_text, example.text
                );
            }
        }
    }
    
    #[test]
    fn test_no_overlapping_entities() {
        for example in all_datasets() {
            let mut spans: Vec<(usize, usize, &str)> = example.entities
                .iter()
                .map(|e| (e.start, e.end, e.text.as_str()))
                .collect();
            spans.sort_by_key(|(start, _, _)| *start);
            
            for window in spans.windows(2) {
                let (_, end1, text1) = window[0];
                let (start2, _, text2) = window[1];
                assert!(
                    end1 <= start2,
                    "Overlapping entities '{}' [{}, {}) and '{}' [{}, {}) in: {}",
                    text1, window[0].0, end1, text2, start2, window[1].1, example.text
                );
            }
        }
    }
}
