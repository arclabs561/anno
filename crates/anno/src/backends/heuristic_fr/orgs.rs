use crate::core::entity::{Entity, EntityType};
use regex::Regex;
use std::sync::OnceLock;

const CONFIDENCE: f32 = 0.85;

fn org_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?:\p{Lu}[\p{L}'\-&]+(?:\s+\p{Lu}[\p{L}'\-&]+){0,4}\s+)(?:Société\s+Civile\s+Immobilière|Société\s+Civile|Auto-?entrepreneur|Micro-?entreprise|SASU|SARL|EURL|SCOP|EPIC|EIRL|SCM|SCP|SCS|SCA|SCI|SEM|GIE|SNC|SAS|SA)\b"
        ).expect("org regex")
    })
}

pub fn extract_orgs(text: &str) -> Vec<Entity> {
    org_re()
        .find_iter(text)
        .map(|m| {
            let start = text[..m.start()].chars().count();
            let end = text[..m.end()].chars().count();
            Entity::builder(m.as_str(), EntityType::Organization)
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
    fn detects_sas() {
        let r = extract_orgs("Acme Tech SAS opère depuis Paris.");
        assert_eq!(r.len(), 1);
        assert!(r[0].text.contains("SAS"));
    }

    #[test]
    fn detects_sarl_and_sci() {
        let r = extract_orgs("Construction Dupont SARL et Patrimoine Familial SCI.");
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn does_not_match_lowercase() {
        let r = extract_orgs("acme tech sas");
        assert!(r.is_empty());
    }

    #[test]
    fn sasu_not_split_as_sa() {
        let r = extract_orgs("Innovate Lab SASU est ici.");
        assert_eq!(r.len(), 1);
        assert!(r[0].text.contains("SASU"));
    }
}
