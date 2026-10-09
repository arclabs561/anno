use crate::core::entity::{Entity, EntityCategory, EntityType};
use regex::Regex;
use std::sync::OnceLock;

fn candidate_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){11,30}\b").expect("iban intl regex")
    })
}

pub fn extract_iban_intl(text: &str) -> Vec<Entity> {
    candidate_re()
        .find_iter(text)
        .filter(|m| crate::pii::iban_mod97_valid(m.as_str()))
        .map(|m| {
            let start = text[..m.start()].chars().count();
            let end = text[..m.end()].chars().count();
            Entity::builder(
                m.as_str(),
                EntityType::Custom {
                    name: "iban".into(),
                    category: EntityCategory::Numeric,
                },
            )
            .span(start, end)
            .confidence(1.0_f32)
            .build()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_german_iban() {
        let r = extract_iban_intl("Virement vers DE89370400440532013000.");
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn detects_french_iban() {
        let r = extract_iban_intl("IBAN : FR1420041010050500013M02606");
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn rejects_wrong_checksum() {
        let r = extract_iban_intl("DE99370400440532013000");
        assert!(r.is_empty());
    }
}
