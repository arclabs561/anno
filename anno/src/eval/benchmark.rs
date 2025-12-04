//! Large-scale benchmark datasets for NER evaluation.
//!
//! Unlike the synthetic module which provides curated examples, this module
//! generates large datasets for stress testing and statistical significance.
//!
//! # Philosophy (burntsushi-style)
//!
//! - **Real** edge cases, not toy examples
//! - Models should perform **just okay** on these, not perfectly
//! - Tests statistical significance, not cherry-picked success cases
//!
//! # Usage
//!
//! ```rust,no_run
//! use anno::eval::benchmark::{generate_large_dataset, EdgeCaseType};
//!
//! // Generate 1000 hard examples
//! let dataset = generate_large_dataset(1000, EdgeCaseType::All);
//! assert!(dataset.len() >= 1000);
//! ```

use super::datasets::GoldEntity;
use super::synthetic::{AnnotatedExample, Difficulty, Domain};
use anno_core::EntityType;

/// Types of edge cases to generate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeCaseType {
    /// All edge case types
    All,
    /// Ambiguous entities (words that could be multiple types)
    Ambiguous,
    /// Unicode edge cases (RTL, combining characters, emoji)
    Unicode,
    /// Dense text (many entities close together)
    Dense,
    /// Sparse text (single entity in long text)
    Sparse,
    /// Nested or overlapping entity candidates
    Nested,
    /// Unusual capitalization or casing
    Casing,
    /// Boundary cases (entities at start/end, with punctuation)
    Boundary,
    /// Multi-word entities (proper name + title, company + suffix)
    MultiWord,
    /// Numeric pattern edge cases
    NumericEdge,
    /// Domain-specific jargon
    Jargon,
}

/// Generate a large dataset of challenging examples.
///
/// Returns at least `min_count` examples, potentially more due to template expansion.
#[must_use]
pub fn generate_large_dataset(
    min_count: usize,
    edge_case_type: EdgeCaseType,
) -> Vec<AnnotatedExample> {
    let mut examples = Vec::with_capacity(min_count);

    match edge_case_type {
        EdgeCaseType::All => {
            let per_type = min_count / 10;
            examples.extend(generate_ambiguous_examples(per_type));
            examples.extend(generate_unicode_examples(per_type));
            examples.extend(generate_dense_examples(per_type));
            examples.extend(generate_sparse_examples(per_type));
            examples.extend(generate_nested_examples(per_type));
            examples.extend(generate_casing_examples(per_type));
            examples.extend(generate_boundary_examples(per_type));
            examples.extend(generate_multiword_examples(per_type));
            examples.extend(generate_numeric_edge_examples(per_type));
            examples.extend(generate_jargon_examples(per_type));
        }
        EdgeCaseType::Ambiguous => examples.extend(generate_ambiguous_examples(min_count)),
        EdgeCaseType::Unicode => examples.extend(generate_unicode_examples(min_count)),
        EdgeCaseType::Dense => examples.extend(generate_dense_examples(min_count)),
        EdgeCaseType::Sparse => examples.extend(generate_sparse_examples(min_count)),
        EdgeCaseType::Nested => examples.extend(generate_nested_examples(min_count)),
        EdgeCaseType::Casing => examples.extend(generate_casing_examples(min_count)),
        EdgeCaseType::Boundary => examples.extend(generate_boundary_examples(min_count)),
        EdgeCaseType::MultiWord => examples.extend(generate_multiword_examples(min_count)),
        EdgeCaseType::NumericEdge => examples.extend(generate_numeric_edge_examples(min_count)),
        EdgeCaseType::Jargon => examples.extend(generate_jargon_examples(min_count)),
    }

    // Ensure we have at least min_count by duplicating with variations
    while examples.len() < min_count {
        let idx = examples.len() % examples.len().max(1);
        if let Some(ex) = examples.get(idx).cloned() {
            examples.push(ex);
        } else {
            break;
        }
    }

    examples
}

// ============================================================================
// Ambiguous Examples - Words that could be multiple types
// ============================================================================

fn generate_ambiguous_examples(count: usize) -> Vec<AnnotatedExample> {
    // Words that are genuinely ambiguous: "Apple" (company or fruit),
    // "Washington" (person, city, or state), "Amazon" (company or river)
    let templates = [
        // Apple - company vs fruit (this is genuinely hard)
        ("I love Apple products but prefer android.", vec![]), // Apple = company, but no clear entity marker
        ("The Apple fell from the tree.", vec![]),             // Apple = fruit, not an entity
        (
            "Apple Inc. reported earnings.",
            vec![("Apple Inc.", EntityType::Organization, 0)],
        ),
        (
            "Steve Jobs founded Apple.",
            vec![
                ("Steve Jobs", EntityType::Person, 0),
                ("Apple", EntityType::Organization, 19),
            ],
        ),
        // Washington - person, city, state, building
        (
            "Washington was the first president.",
            vec![("Washington", EntityType::Person, 0)],
        ),
        (
            "I visited Washington D.C. last week.",
            vec![("Washington D.C.", EntityType::Location, 10)],
        ),
        ("The meeting is at Washington Hotel.", vec![]), // Ambiguous: Location or Organization?
        (
            "George Washington crossed the Delaware.",
            vec![
                ("George Washington", EntityType::Person, 0),
                ("Delaware", EntityType::Location, 30),
            ],
        ),
        // Amazon - company vs river
        (
            "Amazon ships globally.",
            vec![("Amazon", EntityType::Organization, 0)],
        ),
        (
            "The Amazon river is massive.",
            vec![
                ("Amazon", EntityType::Location, 4), // River name
            ],
        ),
        (
            "Amazon Prime is popular.",
            vec![
                ("Amazon Prime", EntityType::Organization, 0), // Product/service
            ],
        ),
        // Jordan - person vs country
        (
            "Michael Jordan played basketball.",
            vec![("Michael Jordan", EntityType::Person, 0)],
        ),
        (
            "Jordan borders Israel.",
            vec![
                ("Jordan", EntityType::Location, 0),
                ("Israel", EntityType::Location, 15),
            ],
        ),
        (
            "Jordan Peele directed the film.",
            vec![("Jordan Peele", EntityType::Person, 0)],
        ),
        // Paris - city vs person (Paris Hilton)
        (
            "Paris is beautiful in spring.",
            vec![("Paris", EntityType::Location, 0)],
        ),
        (
            "Paris Hilton attended the event.",
            vec![("Paris Hilton", EntityType::Person, 0)],
        ),
        // China - country vs dinnerware (lowercase = tableware)
        ("Made in China.", vec![("China", EntityType::Location, 8)]),
        ("The china cabinet is antique.", vec![]), // lowercase china = porcelain
        // May - month vs name vs verb
        (
            "May is a lovely month.",
            vec![
                ("May", EntityType::Date, 0), // Month
            ],
        ),
        (
            "Theresa May resigned as PM.",
            vec![("Theresa May", EntityType::Person, 0)],
        ),
        ("You may proceed.", vec![]), // verb, not entity
        // Bill - name vs legislation vs invoice
        (
            "Bill Gates founded Microsoft.",
            vec![
                ("Bill Gates", EntityType::Person, 0),
                ("Microsoft", EntityType::Organization, 19),
            ],
        ),
        ("The bill passed the Senate.", vec![]), // legislation, not entity
        ("Please pay the bill.", vec![]),        // invoice, not entity
    ];

    generate_from_templates(&templates, count, Domain::News, Difficulty::Hard)
}

// ============================================================================
// Unicode Examples - RTL, combining characters, emoji
// ============================================================================

fn generate_unicode_examples(count: usize) -> Vec<AnnotatedExample> {
    let templates = [
        // CJK characters
        (
            "株式会社トヨタ自動車 reported earnings.",
            vec![("株式会社トヨタ自動車", EntityType::Organization, 0)],
        ),
        (
            "Contact 田中太郎 for details.",
            vec![("田中太郎", EntityType::Person, 8)],
        ),
        // Arabic (RTL)
        (
            "محمد is a common name.",
            vec![("محمد", EntityType::Person, 0)],
        ),
        // Cyrillic
        (
            "Владимир Путин addressed the nation.",
            vec![("Владимир Путин", EntityType::Person, 0)],
        ),
        (
            "Visit Москва this summer.",
            vec![("Москва", EntityType::Location, 6)],
        ),
        // Greek
        (
            "Αθήνα is the capital of Greece.",
            vec![
                ("Αθήνα", EntityType::Location, 0),
                ("Greece", EntityType::Location, 24),
            ],
        ),
        // Mixed scripts
        (
            "Sony (ソニー株式会社) announced new products.",
            vec![
                ("Sony", EntityType::Organization, 0),
                ("ソニー株式会社", EntityType::Organization, 6),
            ],
        ),
        // Emoji with entities
        (
            "Tim Berners-Lee invented the Web.",
            vec![
                ("Tim Berners-Lee", EntityType::Person, 0),
                ("Web", EntityType::Other("Technology".to_string()), 29),
            ],
        ),
        // Accented characters
        (
            "François Hollande was president.",
            vec![("François Hollande", EntityType::Person, 0)],
        ),
        (
            "Zürich is in Switzerland.",
            vec![
                ("Zürich", EntityType::Location, 0),
                ("Switzerland", EntityType::Location, 13),
            ],
        ),
        (
            "São Paulo is Brazil's largest city.",
            vec![
                ("São Paulo", EntityType::Location, 0),
                ("Brazil", EntityType::Location, 13),
            ],
        ),
        // Combining characters
        (
            "José García works at Google.",
            vec![
                ("José García", EntityType::Person, 0),
                ("Google", EntityType::Organization, 21),
            ],
        ),
        // Long multi-byte
        (
            "北京大学 and 清华大学 are top universities.",
            vec![
                ("北京大学", EntityType::Organization, 0),
                ("清华大学", EntityType::Organization, 9),
            ],
        ),
    ];

    generate_from_templates(&templates, count, Domain::News, Difficulty::Hard)
}

// ============================================================================
// Dense Examples - Many entities close together
// ============================================================================

fn generate_dense_examples(count: usize) -> Vec<AnnotatedExample> {
    let templates = [
        // Multiple entities per sentence
        (
            "Apple, Google, Microsoft, Amazon, and Meta all reported earnings.",
            vec![
                ("Apple", EntityType::Organization, 0),
                ("Google", EntityType::Organization, 7),
                ("Microsoft", EntityType::Organization, 15),
                ("Amazon", EntityType::Organization, 26),
                ("Meta", EntityType::Organization, 38),
            ],
        ),
        (
            "John Smith, Jane Doe, and Bob Wilson attended.",
            vec![
                ("John Smith", EntityType::Person, 0),
                ("Jane Doe", EntityType::Person, 12),
                ("Bob Wilson", EntityType::Person, 26),
            ],
        ),
        (
            "Contact support@company.com or sales@company.com.",
            vec![
                ("support@company.com", EntityType::Email, 8),
                ("sales@company.com", EntityType::Email, 31),
            ],
        ),
        (
            "Visit https://example.com and https://test.org.",
            vec![
                ("https://example.com", EntityType::Url, 6),
                ("https://test.org", EntityType::Url, 30),
            ],
        ),
        (
            "$100, $200, $300, and $400 are the prices.",
            vec![
                ("$100", EntityType::Money, 0),
                ("$200", EntityType::Money, 6),
                ("$300", EntityType::Money, 12),
                ("$400", EntityType::Money, 22),
            ],
        ),
        // Very dense entity sequence
        (
            "Tokyo, Beijing, Seoul, Bangkok, Singapore.",
            vec![
                ("Tokyo", EntityType::Location, 0),
                ("Beijing", EntityType::Location, 7),
                ("Seoul", EntityType::Location, 16),
                ("Bangkok", EntityType::Location, 23),
                ("Singapore", EntityType::Location, 32),
            ],
        ),
        // Dates and numbers
        (
            "January 1, February 2, March 3, April 4, May 5.",
            vec![
                ("January 1", EntityType::Date, 0),
                ("February 2", EntityType::Date, 11),
                ("March 3", EntityType::Date, 23),
                ("April 4", EntityType::Date, 32),
                ("May 5", EntityType::Date, 41),
            ],
        ),
        // Mixed entity types, dense
        (
            "Tim Cook (Apple), Sundar Pichai (Google), Satya Nadella (Microsoft).",
            vec![
                ("Tim Cook", EntityType::Person, 0),
                ("Apple", EntityType::Organization, 10),
                ("Sundar Pichai", EntityType::Person, 18),
                ("Google", EntityType::Organization, 33),
                ("Satya Nadella", EntityType::Person, 42),
                ("Microsoft", EntityType::Organization, 57),
            ],
        ),
    ];

    generate_from_templates(&templates, count, Domain::News, Difficulty::Hard)
}

// ============================================================================
// Sparse Examples - Single entity in long text
// ============================================================================

fn generate_sparse_examples(count: usize) -> Vec<AnnotatedExample> {
    let templates = [
        (
            "The weather has been quite nice lately with temperatures hovering around the mid-seventies and clear skies throughout most of the week. According to the forecast, this trend is expected to continue for at least another few days before any significant changes occur in the atmospheric conditions. Many people have been taking advantage of the pleasant weather to spend time outdoors, enjoying activities such as hiking, biking, and picnicking in the parks. The local news station reported that John Smith was seen at the park yesterday.",
            vec![("John Smith", EntityType::Person, 493)],
        ),
        (
            "In a move that surprised absolutely no one, the quarterly financial results came in exactly as analysts had predicted, with revenues matching expectations and profit margins holding steady. The company's stock price barely moved on the news, reflecting the market's general sense that everything was proceeding according to plan. After hours of discussion and deliberation, the board decided to maintain the current dividend policy and continue with the existing strategic initiatives. CEO Maria Garcia addressed the shareholders.",
            vec![("Maria Garcia", EntityType::Person, 490)],
        ),
        (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. The meeting is at Google headquarters.",
            vec![("Google", EntityType::Organization, 210)],
        ),
    ];

    generate_from_templates(&templates, count, Domain::News, Difficulty::Medium)
}

// ============================================================================
// Nested Examples - Overlapping/nested entity candidates
// ============================================================================

fn generate_nested_examples(count: usize) -> Vec<AnnotatedExample> {
    let templates = [
        // Organization within location name
        (
            "University of California, Los Angeles is a great school.",
            vec![(
                "University of California, Los Angeles",
                EntityType::Organization,
                0,
            )],
        ),
        // Person name contains place name
        (
            "Jack London wrote adventure novels.",
            vec![("Jack London", EntityType::Person, 0)],
        ),
        // Location within organization name
        (
            "Bank of America reported earnings.",
            vec![("Bank of America", EntityType::Organization, 0)],
        ),
        // Multiple nested candidates
        (
            "New York Times published the story.",
            vec![("New York Times", EntityType::Organization, 0)],
        ),
        // Person name that is also a place
        (
            "Carolina Herrera designs fashion.",
            vec![("Carolina Herrera", EntityType::Person, 0)],
        ),
        // Compound proper nouns
        (
            "The Wall Street Journal reports on Wall Street.",
            vec![
                ("Wall Street Journal", EntityType::Organization, 4),
                ("Wall Street", EntityType::Location, 35),
            ],
        ),
        // University names (nested location)
        (
            "Harvard University is in Cambridge, Massachusetts.",
            vec![
                ("Harvard University", EntityType::Organization, 0),
                ("Cambridge", EntityType::Location, 25),
                ("Massachusetts", EntityType::Location, 36),
            ],
        ),
    ];

    generate_from_templates(&templates, count, Domain::Academic, Difficulty::Hard)
}

// ============================================================================
// Casing Examples - Unusual capitalization
// ============================================================================

fn generate_casing_examples(count: usize) -> Vec<AnnotatedExample> {
    let templates = [
        // All caps (headlines, tweets)
        (
            "APPLE ANNOUNCES NEW IPHONE",
            vec![("APPLE", EntityType::Organization, 0)],
        ),
        // lowercase (informal text, tweets)
        (
            "just saw tim cook at the apple store lol",
            vec![
                // These should ideally be detected but many models fail here
            ],
        ),
        // CamelCase (product names, hashtags)
        (
            "Download the iPhone app from Apple.",
            vec![("Apple", EntityType::Organization, 29)],
        ),
        // Mixed case (typos, stylistic)
        (
            "BREAKING: Amazon CEO resigns",
            vec![("Amazon", EntityType::Organization, 10)],
        ),
        // Title case vs sentence case
        (
            "The President Of The United States spoke today.",
            vec![("United States", EntityType::Location, 21)],
        ),
        // Acronyms with periods
        (
            "The U.S.A. is large.",
            vec![("U.S.A.", EntityType::Location, 4)],
        ),
        (
            "Contact the F.B.I. for assistance.",
            vec![("F.B.I.", EntityType::Organization, 12)],
        ),
    ];

    generate_from_templates(&templates, count, Domain::SocialMedia, Difficulty::Hard)
}

// ============================================================================
// Boundary Examples - Entities at boundaries, with punctuation
// ============================================================================

fn generate_boundary_examples(count: usize) -> Vec<AnnotatedExample> {
    let templates = [
        // At start
        (
            "Apple is a company.",
            vec![("Apple", EntityType::Organization, 0)],
        ),
        // At end
        (
            "The company is Apple.",
            vec![("Apple", EntityType::Organization, 15)],
        ),
        // With punctuation
        (
            "Is Apple, the company, profitable?",
            vec![("Apple", EntityType::Organization, 3)],
        ),
        // With quotes
        (
            "\"Apple\" announced earnings.",
            vec![("Apple", EntityType::Organization, 1)],
        ),
        // With parentheses
        (
            "The company (Apple) is large.",
            vec![("Apple", EntityType::Organization, 13)],
        ),
        // With colon
        (
            "Company: Apple Inc.",
            vec![("Apple Inc.", EntityType::Organization, 9)],
        ),
        // Hyphenated
        (
            "The Apple-Google partnership.",
            vec![
                ("Apple", EntityType::Organization, 4),
                ("Google", EntityType::Organization, 10),
            ],
        ),
        // With apostrophe
        (
            "Apple's revenue increased.",
            vec![("Apple", EntityType::Organization, 0)],
        ),
        // Entity is the entire text
        ("John Smith", vec![("John Smith", EntityType::Person, 0)]),
        // Entity at very end with period
        (
            "The CEO is Tim Cook.",
            vec![("Tim Cook", EntityType::Person, 11)],
        ),
    ];

    generate_from_templates(&templates, count, Domain::News, Difficulty::Medium)
}

// ============================================================================
// Multi-Word Examples - Complex multi-token entities
// ============================================================================

fn generate_multiword_examples(count: usize) -> Vec<AnnotatedExample> {
    let templates = [
        // Titles with names
        (
            "Dr. John Smith presented the findings.",
            vec![("Dr. John Smith", EntityType::Person, 0)],
        ),
        (
            "Prof. Maria Garcia leads the department.",
            vec![("Prof. Maria Garcia", EntityType::Person, 0)],
        ),
        // Full company names
        (
            "International Business Machines Corporation announced layoffs.",
            vec![(
                "International Business Machines Corporation",
                EntityType::Organization,
                0,
            )],
        ),
        // Locations with qualifiers
        (
            "Greater Los Angeles Area is densely populated.",
            vec![("Greater Los Angeles Area", EntityType::Location, 0)],
        ),
        // Names with suffixes
        (
            "John Smith Jr. inherited the company.",
            vec![("John Smith Jr.", EntityType::Person, 0)],
        ),
        (
            "Robert Kennedy III is running.",
            vec![("Robert Kennedy III", EntityType::Person, 0)],
        ),
        // Organizations with "The"
        (
            "The New York Times reported.",
            vec![("The New York Times", EntityType::Organization, 0)],
        ),
        // Long place names
        (
            "The United Kingdom of Great Britain and Northern Ireland.",
            vec![(
                "United Kingdom of Great Britain and Northern Ireland",
                EntityType::Location,
                4,
            )],
        ),
    ];

    generate_from_templates(&templates, count, Domain::News, Difficulty::Hard)
}

// ============================================================================
// Numeric Edge Examples - Pattern edge cases
// ============================================================================

fn generate_numeric_edge_examples(count: usize) -> Vec<AnnotatedExample> {
    let templates = [
        // Money edge cases
        ("The price is $1.", vec![("$1", EntityType::Money, 13)]),
        ("It costs $0.01.", vec![("$0.01", EntityType::Money, 9)]),
        (
            "Worth $1,000,000.",
            vec![("$1,000,000", EntityType::Money, 6)],
        ),
        ("Price: $1.5M.", vec![("$1.5M", EntityType::Money, 7)]),
        (
            "Costs between $10-$20.",
            vec![
                ("$10", EntityType::Money, 14),
                ("$20", EntityType::Money, 18),
            ],
        ),
        // Percentage edge cases
        ("Increased 0.5%.", vec![("0.5%", EntityType::Percent, 10)]),
        ("Down 100%.", vec![("100%", EntityType::Percent, 5)]),
        ("About 33.333%.", vec![("33.333%", EntityType::Percent, 6)]),
        // Date edge cases
        ("On 1/1/2020.", vec![("1/1/2020", EntityType::Date, 3)]),
        (
            "Date: 2020-01-01.",
            vec![("2020-01-01", EntityType::Date, 6)],
        ),
        (
            "January 1st, 2020.",
            vec![("January 1st, 2020", EntityType::Date, 0)],
        ),
        ("The year 2020.", vec![]), // Year alone may not be a date entity
        // Time edge cases
        ("At 9:00 AM.", vec![("9:00 AM", EntityType::Time, 3)]),
        ("Meeting at 14:30.", vec![("14:30", EntityType::Time, 11)]),
        ("Around noon.", vec![]), // "noon" may or may not be time entity
        // Phone edge cases
        ("Call 555-1234.", vec![("555-1234", EntityType::Phone, 5)]),
        (
            "Phone: +1-800-555-1234.",
            vec![("+1-800-555-1234", EntityType::Phone, 7)],
        ),
        ("Ext. 1234.", vec![]), // Extension alone is not a phone
        // Email edge cases
        ("Email a@b.co.", vec![("a@b.co", EntityType::Email, 6)]),
        (
            "Contact test.user+tag@subdomain.example.com.",
            vec![("test.user+tag@subdomain.example.com", EntityType::Email, 8)],
        ),
        // URL edge cases
        (
            "Visit http://x.co.",
            vec![("http://x.co", EntityType::Url, 6)],
        ),
        ("Go to localhost:8080.", vec![]), // localhost is not typically a URL entity
    ];

    generate_from_templates(&templates, count, Domain::Technical, Difficulty::Hard)
}

// ============================================================================
// Jargon Examples - Domain-specific terminology
// ============================================================================

fn generate_jargon_examples(count: usize) -> Vec<AnnotatedExample> {
    let templates = [
        // Medical jargon - entities hard to distinguish from regular words
        ("The patient has COVID-19.", vec![]), // COVID-19 may be disease, not organization
        (
            "Pfizer developed the vaccine.",
            vec![("Pfizer", EntityType::Organization, 0)],
        ),
        // Legal jargon
        ("Per Brown v. Board of Education.", vec![]), // Case names are complex
        (
            "The defendant, John Smith, pleaded.",
            vec![("John Smith", EntityType::Person, 15)],
        ),
        // Tech jargon - product names vs companies
        ("Install Python 3.10.", vec![]), // Python is language, not entity
        (
            "Microsoft released Windows 11.",
            vec![("Microsoft", EntityType::Organization, 0)],
        ),
        ("The React library is popular.", vec![]), // React is library
        (
            "Facebook created React.",
            vec![("Facebook", EntityType::Organization, 0)],
        ),
        // Financial jargon
        ("S&P 500 rose 2%.", vec![("2%", EntityType::Percent, 13)]), // S&P 500 is index
        ("NASDAQ hit record highs.", vec![]),                        // NASDAQ is index
        (
            "Goldman Sachs upgraded the stock.",
            vec![("Goldman Sachs", EntityType::Organization, 0)],
        ),
        // Sports jargon
        (
            "The Lakers beat the Celtics 110-105.",
            vec![
                ("Lakers", EntityType::Organization, 4),
                ("Celtics", EntityType::Organization, 20),
            ],
        ),
        (
            "LeBron James scored 30 points.",
            vec![("LeBron James", EntityType::Person, 0)],
        ),
    ];

    generate_from_templates(&templates, count, Domain::Technical, Difficulty::Hard)
}

// ============================================================================
// Helper Functions
// ============================================================================

#[allow(clippy::type_complexity)]
fn generate_from_templates(
    templates: &[(&str, Vec<(&str, EntityType, usize)>)],
    count: usize,
    domain: Domain,
    difficulty: Difficulty,
) -> Vec<AnnotatedExample> {
    let mut examples = Vec::with_capacity(count);

    for (text, entity_specs) in templates.iter().cycle().take(count.max(templates.len())) {
        let entities: Vec<GoldEntity> = entity_specs
            .iter()
            .map(|(txt, entity_type, start)| GoldEntity::new(*txt, entity_type.clone(), *start))
            .collect();

        examples.push(AnnotatedExample {
            text: (*text).to_string(),
            entities,
            domain,
            difficulty,
        });
    }

    examples
}

/// Statistics about a benchmark dataset.
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    /// Total number of examples in the dataset.
    pub total_examples: usize,
    /// Total number of entities across all examples.
    pub total_entities: usize,
    /// Average entities per example.
    pub avg_entities_per_example: f64,
    /// Number of examples with no entities (negative examples).
    pub examples_with_no_entities: usize,
    /// Distribution of edge case types (if tracked during generation).
    pub edge_case_distribution: Vec<(EdgeCaseType, usize)>,
}

impl BenchmarkStats {
    /// Calculate stats from a dataset
    pub fn from_dataset(examples: &[AnnotatedExample]) -> Self {
        let total_examples = examples.len();
        let total_entities: usize = examples.iter().map(|e| e.entities.len()).sum();
        let examples_with_no_entities = examples.iter().filter(|e| e.entities.is_empty()).count();

        Self {
            total_examples,
            total_entities,
            avg_entities_per_example: total_entities as f64 / total_examples.max(1) as f64,
            examples_with_no_entities,
            edge_case_distribution: vec![], // Would need tracking during generation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_large_dataset() {
        let dataset = generate_large_dataset(100, EdgeCaseType::All);
        assert!(
            dataset.len() >= 100,
            "Should generate at least 100 examples"
        );
    }

    #[test]
    fn test_ambiguous_examples() {
        let examples = generate_ambiguous_examples(10);
        assert!(!examples.is_empty());
        // Verify some examples have no entities (genuinely ambiguous)
        let no_entity_count = examples.iter().filter(|e| e.entities.is_empty()).count();
        assert!(
            no_entity_count > 0,
            "Should have some ambiguous (no entity) cases"
        );
    }

    #[test]
    fn test_unicode_examples() {
        let examples = generate_unicode_examples(10);
        assert!(!examples.is_empty());
        // Verify we have non-ASCII text
        let has_unicode = examples
            .iter()
            .any(|e| e.text.chars().any(|c| !c.is_ascii()));
        assert!(has_unicode, "Should have non-ASCII characters");
    }

    #[test]
    fn test_entity_offsets_valid() {
        let dataset = generate_large_dataset(500, EdgeCaseType::All);
        for example in &dataset {
            for entity in &example.entities {
                assert!(
                    entity.start < example.text.len(),
                    "Entity '{}' start {} >= text length {} in: {}",
                    entity.text,
                    entity.start,
                    example.text.len(),
                    example.text
                );
                // Note: Using chars().count() for Unicode-safe comparison
                let char_count = example.text.chars().count();
                assert!(
                    entity.end <= char_count,
                    "Entity '{}' end {} > char count {} in: {}",
                    entity.text,
                    entity.end,
                    char_count,
                    example.text
                );
            }
        }
    }

    #[test]
    fn test_benchmark_stats() {
        let dataset = generate_large_dataset(100, EdgeCaseType::All);
        let stats = BenchmarkStats::from_dataset(&dataset);
        assert!(stats.total_examples >= 100);
        assert!(stats.total_entities > 0);
    }

    // Slow test - generates many examples
    #[test]
    #[ignore] // Run with: cargo test -- --ignored
    fn test_large_scale_benchmark() {
        let dataset = generate_large_dataset(10_000, EdgeCaseType::All);
        assert!(dataset.len() >= 10_000);
        let stats = BenchmarkStats::from_dataset(&dataset);
        println!("Large benchmark stats: {:?}", stats);
    }
}
