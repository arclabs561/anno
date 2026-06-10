use crate::core::entity::{Entity, EntityCategory, EntityType};
use regex::Regex;
use std::sync::OnceLock;

const DOB_TRIGGERS: &[&str] = &[
    "né le",
    "née le",
    "né(e) le",
    "date de naissance",
    "naissance :",
    "naissance:",
    "anniversaire",
    "âgé",
];
const WINDOW: usize = 50;

fn date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|décembre)\s+\d{4}|\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b"
        ).expect("date regex")
    })
}

pub fn extract_dates(text: &str) -> Vec<Entity> {
    date_re()
        .find_iter(text)
        .map(|m| {
            let is_dob = has_dob_trigger(text, m.start());
            let label = if is_dob {
                EntityType::Custom {
                    name: "date_of_birth".into(),
                    category: EntityCategory::Temporal,
                }
            } else {
                EntityType::Date
            };
            let start = text[..m.start()].chars().count();
            let end = text[..m.end()].chars().count();
            Entity::builder(m.as_str(), label)
                .span(start, end)
                .confidence(0.90_f32)
                .build()
        })
        .collect()
}

fn has_dob_trigger(text: &str, byte_idx: usize) -> bool {
    let start = byte_idx.saturating_sub(WINDOW);
    let snippet = text[start..byte_idx].to_lowercase();
    DOB_TRIGGERS.iter().any(|t| snippet.contains(t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_textual_date() {
        let r = extract_dates("Réunion fixée au 15 janvier 2024.");
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn detects_numeric_date() {
        let r = extract_dates("Échéance : 31/12/2025.");
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn relabels_dob_with_trigger() {
        let r = extract_dates("M. Dupont, né le 12 mai 1972.");
        assert_eq!(r.len(), 1);
        assert!(
            matches!(&r[0].entity_type, EntityType::Custom { name, .. } if name == "date_of_birth")
        );
    }

    #[test]
    fn plain_date_without_trigger() {
        let r = extract_dates("Le contrat prend effet le 1 mars 2024.");
        assert!(matches!(r[0].entity_type, EntityType::Date));
    }
}
