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
        .filter(|m| iban_mod97(m.as_str()))
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

fn iban_mod97(raw: &str) -> bool {
    let s: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    if s.len() < 15 || s.len() > 34 {
        return false;
    }
    let s = s.to_ascii_uppercase();
    if !s.chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    let (head, tail) = s.split_at(4);
    let rearranged = format!("{tail}{head}");
    let mut numeric = String::with_capacity(rearranged.len() * 2);
    for c in rearranged.chars() {
        if c.is_ascii_digit() {
            numeric.push(c);
        } else {
            numeric.push_str(&((c as u32 - 'A' as u32 + 10).to_string()));
        }
    }
    let mut r: u32 = 0;
    for d in numeric.chars() {
        r = (r * 10 + d.to_digit(10).unwrap()) % 97;
    }
    r == 1
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
