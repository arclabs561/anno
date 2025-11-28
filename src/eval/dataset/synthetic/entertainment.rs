//! Entertainment domain synthetic data (movies, music, celebrities).

use super::super::types::helpers::entity;
use super::super::types::{AnnotatedExample, Difficulty, Domain};
use crate::EntityType;

/// Entertainment domain dataset (movies, music, celebrities).
pub fn dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Tom Hanks won an Oscar for his role in Forrest Gump directed by Robert Zemeckis."
                .into(),
            entities: vec![
                entity("Tom Hanks", EntityType::Person, 0),
                entity("Robert Zemeckis", EntityType::Person, 64),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Taylor Swift's Eras Tour broke records at SoFi Stadium in Los Angeles.".into(),
            entities: vec![
                entity("Taylor Swift", EntityType::Person, 0),
                entity("SoFi Stadium", EntityType::Location, 42),
                entity("Los Angeles", EntityType::Location, 58),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Disney acquired 21st Century Fox for $71 billion in 2019.".into(),
            entities: vec![
                entity("Disney", EntityType::Organization, 0),
                entity("21st Century Fox", EntityType::Organization, 16),
                entity("$71 billion", EntityType::Money, 37),
                entity("2019", EntityType::Date, 52),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Beyoncé performed at Coachella in Indio, California in April 2018.".into(),
            entities: vec![
                entity("Beyoncé", EntityType::Person, 0),
                entity("Indio", EntityType::Location, 34),
                entity("California", EntityType::Location, 41),
                entity("April 2018", EntityType::Date, 55),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text: "Netflix released Stranger Things season 4 starring Millie Bobby Brown.".into(),
            entities: vec![
                entity("Netflix", EntityType::Organization, 0),
                entity("Millie Bobby Brown", EntityType::Person, 51),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "The Beatles recorded Abbey Road at Abbey Road Studios in London.".into(),
            entities: vec![
                entity("Abbey Road Studios", EntityType::Organization, 35),
                entity("London", EntityType::Location, 57),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Hard,
        },
        AnnotatedExample {
            text: "Christopher Nolan directed Oppenheimer starring Cillian Murphy.".into(),
            entities: vec![
                entity("Christopher Nolan", EntityType::Person, 0),
                entity("Cillian Murphy", EntityType::Person, 48),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Easy,
        },
        AnnotatedExample {
            text: "BTS announced their hiatus in June 2022 before members enlisted in South Korea."
                .into(),
            entities: vec![
                entity("June 2022", EntityType::Date, 30),
                entity("South Korea", EntityType::Location, 67),
            ],
            domain: Domain::Entertainment,
            difficulty: Difficulty::Medium,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entertainment_dataset_not_empty() {
        assert!(!dataset().is_empty());
    }
}

