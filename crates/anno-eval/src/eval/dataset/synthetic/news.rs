//! News domain synthetic data (CoNLL-2003 style).

use super::super::types::helpers::entity;
use super::super::types::{AnnotatedExample, Difficulty, Domain};
use anno_core::EntityType;

/// News domain dataset (CoNLL-2003 style).
pub fn dataset() -> Vec<AnnotatedExample> {
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
            text: "According to Reuters, Nvidia's Jensen Huang met with German Chancellor Olaf Scholz in Berlin.".into(),
            entities: vec![
                entity("Reuters", EntityType::Organization, 13),
                entity("Nvidia", EntityType::Organization, 22),
                entity("Jensen Huang", EntityType::Person, 31),
                entity("Olaf Scholz", EntityType::Person, 71),
                entity("Berlin", EntityType::Location, 86),
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
        AnnotatedExample {
            text: "Amazon Web Services announced expansion plans in Tokyo and Singapore.".into(),
            entities: vec![
                entity("Amazon Web Services", EntityType::Organization, 0),
                entity("Tokyo", EntityType::Location, 49),
                entity("Singapore", EntityType::Location, 59),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "The United Nations held climate talks in Paris last December.".into(),
            entities: vec![
                entity("United Nations", EntityType::Organization, 4),
                entity("Paris", EntityType::Location, 41),
                entity("December", EntityType::Date, 52),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Warren Buffett's Berkshire Hathaway reported $30 billion in earnings.".into(),
            entities: vec![
                entity("Warren Buffett", EntityType::Person, 0),
                entity("Berkshire Hathaway", EntityType::Organization, 17),
                entity("$30 billion", EntityType::Money, 45),
            ],
            domain: Domain::News,
            difficulty: Difficulty::Medium,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_news_dataset_not_empty() {
        assert!(!dataset().is_empty());
    }

    #[test]
    fn test_all_news_domain() {
        for ex in dataset() {
            assert_eq!(ex.domain, Domain::News);
        }
    }
}
