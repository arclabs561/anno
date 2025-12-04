//! Scientific/academic domain synthetic data.

use super::super::types::helpers::entity;
use super::super::types::{AnnotatedExample, Difficulty, Domain};
use anno_core::EntityType;

/// Scientific/academic domain dataset.
pub fn dataset() -> Vec<AnnotatedExample> {
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
        AnnotatedExample {
            text: "NASA launched Artemis from Cape Canaveral, Florida on November 16, 2022.".into(),
            entities: vec![
                entity("NASA", EntityType::Organization, 0),
                entity("Cape Canaveral", EntityType::Location, 27),
                entity("Florida", EntityType::Location, 43),
                entity("November 16, 2022", EntityType::Date, 54),
            ],
            domain: Domain::Scientific,
            difficulty: Difficulty::Medium,
        },
        AnnotatedExample {
            text:
                "Einstein published special relativity while working at the Swiss Patent Office in Bern."
                    .into(),
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
            text: "The Voyager 1 spacecraft, launched in 1977, is now 15 billion miles from Earth."
                .into(),
            entities: vec![entity("Earth", EntityType::Location, 73)],
            domain: Domain::Scientific,
            difficulty: Difficulty::Hard,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scientific_dataset_not_empty() {
        assert!(!dataset().is_empty());
    }
}
