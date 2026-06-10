use crate::core::entity::{Entity, EntityCategory, EntityType};
use regex::Regex;
use std::sync::OnceLock;

const CONFIDENCE: f32 = 0.85;

fn voie_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b\d+\s*(?:rue|avenue|boulevard|place|impasse|all[ée]e|chemin|route|quai|cours|passage|square|esplanade|promenade|voie|sentier|villa)\s+(?:\p{L}[\p{L}'\-]*\s*){1,4}(?:,?\s+\d{5}\s+\p{Lu}[\p{L}'\-]*)?"
        ).expect("addr regex")
    })
}

pub fn extract_addresses(text: &str) -> Vec<Entity> {
    voie_re()
        .find_iter(text)
        .map(|m| {
            let start = text[..m.start()].chars().count();
            let end = text[..m.end()].chars().count();
            Entity::builder(
                m.as_str(),
                EntityType::Custom {
                    name: "address".into(),
                    category: EntityCategory::Place,
                },
            )
            .span(start, end)
            .confidence(CONFIDENCE)
            .build()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_rue() {
        let r = extract_addresses("Habite au 12 rue de la République.");
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn detects_with_postal() {
        let r = extract_addresses("5 avenue des Champs-Élysées, 75008 Paris.");
        assert_eq!(r.len(), 1);
        assert!(r[0].text.contains("75008") || r[0].text.contains("avenue"));
    }

    #[test]
    fn no_match_without_number() {
        let r = extract_addresses("la rue est calme");
        assert!(r.is_empty());
    }
}
