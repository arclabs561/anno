//! Social media domain synthetic data (WNUT style - noisy text).

use super::super::types::helpers::{entity, entity_url};
use super::super::types::{AnnotatedExample, Difficulty, Domain};
use anno_core::EntityType;

/// Social media dataset (WNUT style - noisy text).
pub fn dataset() -> Vec<AnnotatedExample> {
    vec![
        AnnotatedExample {
            text: "Just saw @satlonapatel at Nvidia HQ in Santa Clara!".into(),
            entities: vec![
                entity("Nvidia", EntityType::Organization, 26),
                entity("Santa Clara", EntityType::Location, 39),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_social_media_dataset_not_empty() {
        assert!(!dataset().is_empty());
    }

    #[test]
    fn test_all_social_media_domain() {
        for ex in dataset() {
            assert_eq!(ex.domain, Domain::SocialMedia);
        }
    }
}
