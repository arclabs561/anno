//! Miscellaneous synthetic datasets: adversarial, structured, conversational, historical.

use super::super::types::helpers::entity;
use super::super::types::{AnnotatedExample, Difficulty, Domain};
use crate::EntityType;

/// Adversarial and edge case examples.
///
/// These examples test challenging scenarios:
/// - Ambiguous entities
/// - Nested entities
/// - Unicode names
/// - Unusual capitalization
/// - Empty/negative examples
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
        // Entity at very start
        AnnotatedExample {
            text: "Microsoft announced earnings.".into(),
            entities: vec![entity("Microsoft", EntityType::Organization, 0)],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Entity at very end
        AnnotatedExample {
            text: "The CEO is Tim Cook".into(),
            entities: vec![entity("Tim Cook", EntityType::Person, 11)],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Multiple adjacent entities
        AnnotatedExample {
            text: "John Smith Mary Jones met in Paris France.".into(),
            entities: vec![
                entity("John Smith", EntityType::Person, 0),
                entity("Mary Jones", EntityType::Person, 11),
                entity("Paris", EntityType::Location, 29),
                entity("France", EntityType::Location, 35),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Same text, different types
        AnnotatedExample {
            text: "Washington visited Washington to meet Washington.".into(),
            entities: vec![
                entity("Washington", EntityType::Person, 0),
                entity("Washington", EntityType::Location, 19),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Very long entity
        AnnotatedExample {
            text: "The International Business Machines Corporation announced results.".into(),
            entities: vec![entity(
                "International Business Machines Corporation",
                EntityType::Organization,
                4,
            )],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Hyphenated name
        AnnotatedExample {
            text: "Mary-Jane Watson works at Stark-Industries.".into(),
            entities: vec![
                entity("Mary-Jane Watson", EntityType::Person, 0),
                entity("Stark-Industries", EntityType::Organization, 26),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Possessive form
        AnnotatedExample {
            text: "Nvidia's Jensen Huang spoke at Google's headquarters.".into(),
            entities: vec![
                entity("Nvidia", EntityType::Organization, 0),
                entity("Jensen Huang", EntityType::Person, 9),
                entity("Google", EntityType::Organization, 31),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Numeric in name
        AnnotatedExample {
            text: "7-Eleven partnered with 3M Corporation.".into(),
            entities: vec![
                entity("7-Eleven", EntityType::Organization, 0),
                entity("3M Corporation", EntityType::Organization, 24),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // All caps
        AnnotatedExample {
            text: "NASA and IBM announced a partnership with MIT.".into(),
            entities: vec![
                entity("NASA", EntityType::Organization, 0),
                entity("IBM", EntityType::Organization, 9),
                entity("MIT", EntityType::Organization, 42),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Adversarial,
        },
        // Abbreviation vs full name
        AnnotatedExample {
            text: "The UN (United Nations) met in NYC (New York City).".into(),
            entities: vec![
                entity("UN", EntityType::Organization, 4),
                entity("United Nations", EntityType::Organization, 8),
                entity("NYC", EntityType::Location, 31),
                entity("New York City", EntityType::Location, 36),
            ],
            domain: Domain::Politics,
            difficulty: Difficulty::Adversarial,
        },
        // Code-mixed text (English-Spanish)
        AnnotatedExample {
            text: "María went to Los Angeles para visitar a Juan.".into(),
            entities: vec![
                entity("María", EntityType::Person, 0),
                entity("Los Angeles", EntityType::Location, 14),
                entity("Juan", EntityType::Person, 41),
            ],
            domain: Domain::SocialMedia,
            difficulty: Difficulty::Adversarial,
        },
    ]
}

/// Structured entities dataset - dates, times, money, percentages.
///
/// These are entities that RegexNER can reliably detect via regex patterns.
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
        // Percentages
        AnnotatedExample {
            text: "Sales increased by 25% while costs dropped 10%.".into(),
            entities: vec![
                entity("25%", EntityType::Percent, 19),
                entity("10%", EntityType::Percent, 43),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The approval rating is 45.5 percent among voters.".into(),
            entities: vec![entity("45.5 percent", EntityType::Percent, 23)],
            domain: Domain::Politics,
            difficulty: Difficulty::Medium,
        },
    ]
}

/// Conversational/dialogue dataset.
pub fn conversational_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Hey, did you hear about John's promotion at Google?".into(),
            entities: vec![
                entity("John", EntityType::Person, 24),
                entity("Google", EntityType::Organization, 44),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "I'm meeting Sarah in New York next Tuesday.".into(),
            entities: vec![
                entity("Sarah", EntityType::Person, 12),
                entity("New York", EntityType::Location, 21),
                entity("Tuesday", EntityType::Date, 35),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "Can you believe Microsoft paid $75 billion for Activision?".into(),
            entities: vec![
                entity("Microsoft", EntityType::Organization, 16),
                entity("$75 billion", EntityType::Money, 31),
                entity("Activision", EntityType::Organization, 47),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "My friend works at Amazon's office in Seattle.".into(),
            entities: vec![
                entity("Amazon", EntityType::Organization, 19),
                entity("Seattle", EntityType::Location, 38),
            ],
            domain: Domain::Conversational,
            difficulty: Difficulty::Easy,
        },
    ]
}

/// Historical text dataset.
pub fn historical_dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Napoleon Bonaparte was exiled to Saint Helena after the Battle of Waterloo in 1815."
                .into(),
            entities: vec![
                entity("Napoleon Bonaparte", EntityType::Person, 0),
                entity("Saint Helena", EntityType::Location, 33),
                entity("1815", EntityType::Date, 78),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Queen Victoria ruled the British Empire from 1837 to 1901.".into(),
            entities: vec![
                entity("Queen Victoria", EntityType::Person, 0),
                entity("British Empire", EntityType::Organization, 25),
                entity("1837", EntityType::Date, 45),
                entity("1901", EntityType::Date, 53),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The Roman Empire fell in 476 AD when Romulus Augustulus was deposed.".into(),
            entities: vec![
                entity("Roman Empire", EntityType::Organization, 4),
                entity("476 AD", EntityType::Date, 25),
                entity("Romulus Augustulus", EntityType::Person, 37),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Abraham Lincoln delivered the Gettysburg Address on November 19, 1863.".into(),
            entities: vec![
                entity("Abraham Lincoln", EntityType::Person, 0),
                entity("November 19, 1863", EntityType::Date, 52),
            ],
            domain: Domain::Historical,
            difficulty: Difficulty::Easy,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adversarial_dataset_not_empty() {
        assert!(!adversarial_dataset().is_empty());
    }

    #[test]
    fn test_structured_dataset_not_empty() {
        assert!(!structured_dataset().is_empty());
    }

    #[test]
    fn test_conversational_dataset_not_empty() {
        assert!(!conversational_dataset().is_empty());
    }

    #[test]
    fn test_historical_dataset_not_empty() {
        assert!(!historical_dataset().is_empty());
    }
}
