//! Legal/regulatory domain synthetic data.

use super::super::types::helpers::{entity, entity_email, entity_phone};
use super::super::types::{AnnotatedExample, Difficulty, Domain};
use crate::EntityType;

/// Legal/regulatory domain dataset.
pub fn dataset() -> Vec<AnnotatedExample> {
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
            text:
                "Judge Ketanji Brown Jackson was confirmed to the Supreme Court on April 7, 2022."
                    .into(),
            entities: vec![
                entity("Ketanji Brown Jackson", EntityType::Person, 6),
                entity("Supreme Court", EntityType::Organization, 49),
                entity("April 7, 2022", EntityType::Date, 66),
            ],
            domain: Domain::Legal,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The FTC filed an antitrust case against Meta Platforms in Washington D.C."
                .into(),
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
            text: "The SEC charged Sam Bankman-Fried with securities fraud totaling $8 billion."
                .into(),
            entities: vec![
                entity("SEC", EntityType::Organization, 4),
                entity("Sam Bankman-Fried", EntityType::Person, 16),
                entity("$8 billion", EntityType::Money, 65),
            ],
            domain: Domain::Legal,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Contact our firm at legal@lawpartners.com or (212) 555-1234 for a consultation."
                .into(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legal_dataset_not_empty() {
        assert!(!dataset().is_empty());
    }
}
