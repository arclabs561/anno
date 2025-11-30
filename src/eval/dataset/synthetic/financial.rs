//! Financial/business domain synthetic data.

use super::super::types::helpers::entity;
use super::super::types::{AnnotatedExample, Difficulty, Domain};
use crate::EntityType;

/// Financial/business domain dataset.
pub fn dataset() -> Vec<AnnotatedExample> {
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
        AnnotatedExample {
            text: "BlackRock manages over $10 trillion in assets globally.".into(),
            entities: vec![
                entity("BlackRock", EntityType::Organization, 0),
                entity("$10 trillion", EntityType::Money, 23),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "JPMorgan Chase CEO Jamie Dimon warned about recession risks.".into(),
            entities: vec![
                entity("JPMorgan Chase", EntityType::Organization, 0),
                entity("Jamie Dimon", EntityType::Person, 19),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The S&P 500 closed at 4,500 points, up 2.3% for the week.".into(),
            entities: vec![entity("2.3%", EntityType::Percent, 39)],
            domain: Domain::Financial,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Visa and Mastercard processed $15 trillion in transactions in 2023.".into(),
            entities: vec![
                entity("Visa", EntityType::Organization, 0),
                entity("Mastercard", EntityType::Organization, 9),
                entity("$15 trillion", EntityType::Money, 30),
                entity("2023", EntityType::Date, 62),
            ],
            domain: Domain::Financial,
            difficulty: Difficulty::Medium,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_financial_dataset_not_empty() {
        assert!(!dataset().is_empty());
    }
}
