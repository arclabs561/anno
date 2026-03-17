//! PII (personally identifiable information) detection and redaction.
//!
//! Two detection paths:
//! - [`classify_entity`]: classifies NER entities as PII (uses character offsets from `Entity`)
//! - [`scan_patterns`]: regex-based pre-NER scan for structured PII (SSN, credit card, IBAN, email, phone, address)
//!
//! After detection, use [`redact`] or [`pseudonymize`] to produce sanitized text.
//!
//! # Example
//!
//! ```
//! use anno::{Model, StackedNER};
//! use anno::pii;
//!
//! let m = StackedNER::default();
//! let text = "John Smith's SSN is 123-45-6789.";
//! let ents = m.extract_entities(text, None)?;
//!
//! // Classify NER entities as PII
//! let mut pii_entities: Vec<pii::PiiEntity> = ents.iter().filter_map(pii::classify_entity).collect();
//! // Also scan for structured PII patterns
//! pii_entities.extend(pii::scan_patterns(text));
//!
//! let report = pii::report(&pii_entities);
//! let redacted = pii::redact(text, &pii_entities);
//! # Ok::<(), anno::Error>(())
//! ```

use anno_core::Entity;
use regex::Regex;
use std::collections::HashMap;

/// A detected PII entity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PiiEntity {
    /// The PII text.
    pub text: String,
    /// Type of PII: `PERSON`, `DOB`, `ADDRESS`, `CONTACT`, `ID_NUMBER`.
    pub pii_type: String,
    /// Start character offset.
    pub start: usize,
    /// End character offset (exclusive).
    pub end: usize,
    /// Risk level: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.
    pub risk_level: String,
}

/// Summary of PII found in text.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PiiReport {
    /// Count of person name entities.
    pub person_count: usize,
    /// Count of date/time entities (potential DOBs).
    pub date_count: usize,
    /// Count of address entities.
    pub location_count: usize,
    /// Count of contact info (email, phone).
    pub contact_count: usize,
    /// Count of ID numbers (SSN, credit card, IBAN).
    pub id_number_count: usize,
    /// All detected PII entities.
    pub entities: Vec<PiiEntity>,
    /// k-anonymity risk assessment.
    pub k_anonymity_risk: String,
}

/// Classify an NER entity as PII.
///
/// Returns `None` if the entity is not PII (e.g., regular dates, general locations).
pub fn classify_entity(entity: &Entity) -> Option<PiiEntity> {
    let label = entity.entity_type.as_label();
    let text = &entity.text;

    let (pii_type, risk_level) = match label {
        "PER" | "PERSON" => ("PERSON", assess_person_risk(text)),
        "DATE" => {
            if looks_like_dob(text) {
                ("DOB", "HIGH")
            } else {
                return None;
            }
        }
        "LOC" | "GPE" | "LOCATION" => {
            if looks_like_address(text) {
                ("ADDRESS", "HIGH")
            } else {
                return None;
            }
        }
        "EMAIL" => ("CONTACT", "HIGH"),
        "PHONE" => ("CONTACT", "HIGH"),
        "URL" | "MONEY" => return None,
        _ => {
            if looks_like_id_number(text) {
                ("ID_NUMBER", "CRITICAL")
            } else {
                return None;
            }
        }
    };

    Some(PiiEntity {
        text: text.clone(),
        pii_type: pii_type.to_string(),
        start: entity.start(),
        end: entity.end(),
        risk_level: risk_level.to_string(),
    })
}

/// Scan text for structured PII patterns (SSN, credit card, IBAN, email, phone, address).
///
/// This is independent of NER -- it catches structured PII via regex.
/// Offsets are character offsets (Unicode scalar values), consistent with [`classify_entity`].
pub fn scan_patterns(text: &str) -> Vec<PiiEntity> {
    let mut results = Vec::new();

    let patterns: &[(&str, &str, &str)] = &[
        (r"\b\d{3}-\d{2}-\d{4}\b", "ID_NUMBER", "CRITICAL"),
        (
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "ID_NUMBER",
            "CRITICAL",
        ),
        (
            r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]{0,16})?\b",
            "ID_NUMBER",
            "CRITICAL",
        ),
        (
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "CONTACT",
            "HIGH",
        ),
        (
            r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "CONTACT",
            "HIGH",
        ),
        (
            r"\b\d{1,5}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Circle|Cir|Terrace|Ter)\.?(?:,\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?\b",
            "ADDRESS",
            "HIGH",
        ),
    ];

    for &(pat, pii_type, risk) in patterns {
        if let Ok(re) = Regex::new(pat) {
            for m in re.find_iter(text) {
                // Convert byte offsets from regex to character offsets
                let start = text[..m.start()].chars().count();
                let end = text[..m.end()].chars().count();
                let overlaps = results
                    .iter()
                    .any(|e: &PiiEntity| !(end <= e.start || start >= e.end));
                if !overlaps {
                    results.push(PiiEntity {
                        text: m.as_str().to_string(),
                        pii_type: pii_type.to_string(),
                        start,
                        end,
                        risk_level: risk.to_string(),
                    });
                }
            }
        }
    }

    results
}

/// Generate a PII report from detected entities.
pub fn report(entities: &[PiiEntity]) -> PiiReport {
    let mut person_count = 0;
    let mut date_count = 0;
    let mut location_count = 0;
    let mut contact_count = 0;
    let mut id_number_count = 0;

    for e in entities {
        match e.pii_type.as_str() {
            "PERSON" => person_count += 1,
            "DOB" => date_count += 1,
            "ADDRESS" => location_count += 1,
            "CONTACT" => contact_count += 1,
            "ID_NUMBER" => id_number_count += 1,
            _ => {}
        }
    }

    let unique_names: std::collections::HashSet<_> = entities
        .iter()
        .filter(|e| e.pii_type == "PERSON")
        .map(|e| e.text.to_lowercase())
        .collect();

    let k_anonymity_risk = if id_number_count > 0 {
        "CRITICAL (direct identifiers present)"
    } else if unique_names.len() > 5 && date_count > 0 && location_count > 0 {
        "HIGH (quasi-identifier combination)"
    } else if unique_names.len() > 3 {
        "MEDIUM (multiple names)"
    } else {
        "LOW"
    };

    PiiReport {
        person_count,
        date_count,
        location_count,
        contact_count,
        id_number_count,
        entities: entities.to_vec(),
        k_anonymity_risk: k_anonymity_risk.to_string(),
    }
}

/// Redact PII by replacing with type tokens (`[PERSON_1]`, `[ID_NUMBER_2]`, etc.).
///
/// Entity offsets are character offsets (Unicode scalar values).
pub fn redact(text: &str, entities: &[PiiEntity]) -> String {
    let mut result = text.to_string();
    let mut type_counts: HashMap<&str, usize> = HashMap::new();

    let mut sorted: Vec<_> = entities.iter().collect();
    sorted.sort_by(|a, b| b.start.cmp(&a.start));

    for entity in sorted {
        let count = type_counts.entry(&entity.pii_type).or_insert(0);
        *count += 1;
        let replacement = format!("[{}_{}]", entity.pii_type, count);
        // Convert char offsets to byte offsets for replace_range
        let byte_start: usize = result
            .chars()
            .take(entity.start)
            .map(|c| c.len_utf8())
            .sum();
        let byte_end: usize = result.chars().take(entity.end).map(|c| c.len_utf8()).sum();
        result.replace_range(byte_start..byte_end, &replacement);
    }

    result
}

/// Pseudonymize PII with consistent fake values.
///
/// Returns `(pseudonymized_text, mapping)` where mapping maps original -> fake
/// for audit/re-identification purposes.
pub fn pseudonymize(text: &str, entities: &[PiiEntity]) -> (String, HashMap<String, String>) {
    let mut result = text.to_string();
    let mut mapping: HashMap<String, String> = HashMap::new();
    let mut name_counter = 0;
    let mut date_counter = 0;
    let mut addr_counter = 0;

    let fake_names = [
        "John Smith",
        "Jane Doe",
        "Alex Johnson",
        "Sam Williams",
        "Chris Brown",
        "Pat Davis",
        "Jordan Miller",
        "Taylor Wilson",
        "Morgan Lee",
        "Casey Martinez",
    ];

    let mut sorted: Vec<_> = entities.iter().collect();
    sorted.sort_by(|a, b| b.start.cmp(&a.start));

    for entity in sorted {
        let fake = if let Some(existing) = mapping.get(&entity.text) {
            existing.clone()
        } else {
            let fake = match entity.pii_type.as_str() {
                "PERSON" => {
                    let name = fake_names[name_counter % fake_names.len()];
                    name_counter += 1;
                    name.to_string()
                }
                "DOB" => {
                    date_counter += 1;
                    format!("1990-01-{:02}", (date_counter % 28) + 1)
                }
                "ADDRESS" => {
                    addr_counter += 1;
                    format!("{} Main St", 100 + addr_counter)
                }
                "CONTACT" => {
                    if entity.text.contains('@') {
                        "contact@example.com".to_string()
                    } else {
                        format!("555-000-{:04}", (entity.start % 9000) + 1000)
                    }
                }
                "ID_NUMBER" => "XXX-XX-XXXX".to_string(),
                _ => "[REDACTED]".to_string(),
            };
            mapping.insert(entity.text.clone(), fake.clone());
            fake
        };

        // Convert char offsets to byte offsets for replace_range
        let byte_start: usize = result
            .chars()
            .take(entity.start)
            .map(|c| c.len_utf8())
            .sum();
        let byte_end: usize = result.chars().take(entity.end).map(|c| c.len_utf8()).sum();
        result.replace_range(byte_start..byte_end, &fake);
    }

    (result, mapping)
}

/// Scan for PII and redact in one call.
///
/// Combines [`classify_entity`] (NER-based) with [`scan_patterns`] (regex-based)
/// and applies [`redact`].
///
/// ```
/// use anno::{pii, Model, StackedNER};
///
/// let text = "John's SSN is 123-45-6789.";
/// let m = StackedNER::default();
/// let redacted = pii::scan_and_redact(text, &m)?;
/// assert!(!redacted.contains("123-45-6789"));
/// # Ok::<(), anno::Error>(())
/// ```
pub fn scan_and_redact(text: &str, model: &dyn crate::Model) -> crate::Result<String> {
    let entities = model.extract_entities(text, None)?;
    let mut pii_entities: Vec<PiiEntity> = entities.iter().filter_map(classify_entity).collect();
    pii_entities.extend(scan_patterns(text));
    Ok(redact(text, &pii_entities))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn assess_person_risk(text: &str) -> &'static str {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= 3 {
        "HIGH"
    } else if words.len() == 2 {
        "MEDIUM"
    } else {
        "LOW"
    }
}

fn looks_like_dob(text: &str) -> bool {
    Regex::new(r"19[0-9]{2}|20[0-1][0-9]")
        .map(|re| re.is_match(text))
        .unwrap_or(false)
}

/// Check if text looks like a physical address.
pub fn looks_like_address(text: &str) -> bool {
    let has_number = text.chars().any(|c| c.is_numeric());
    let street_indicators = [
        "St", "Street", "Ave", "Avenue", "Rd", "Road", "Blvd", "Dr", "Lane", "Ln", "Way", "Drive",
        "Court", "Ct", "Place", "Pl", "Circle", "Cir",
    ];
    let has_street = street_indicators.iter().any(|ind| text.contains(ind));

    let has_zip = Regex::new(r"\b\d{5}(?:-\d{4})?\b")
        .map(|re| re.is_match(text))
        .unwrap_or(false);
    let us_states = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
        "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
        "VA", "WA", "WV", "WI", "WY", "DC",
    ];
    let has_state = us_states.iter().any(|s| text.contains(s));

    (has_number && has_street) || (has_zip && has_state)
}

/// Check if text looks like an ID number (SSN, credit card, IBAN, MRN).
pub fn looks_like_id_number(text: &str) -> bool {
    if let Ok(re) = Regex::new(r"\d{3}-\d{2}-\d{4}") {
        if re.is_match(text) {
            return true;
        }
    }
    if let Ok(re) = Regex::new(r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}") {
        if re.is_match(text) {
            return true;
        }
    }
    if let Ok(re) = Regex::new(r"[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]{0,16})?") {
        if re.is_match(text) {
            return true;
        }
    }
    if text.len() >= 6
        && text.len() <= 10
        && text.chars().all(|c| c.is_alphanumeric())
        && text.chars().any(|c| c.is_ascii_digit())
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssn_detected_by_scan() {
        let pii = scan_patterns("My SSN is 123-45-6789 and that's it.");
        assert!(pii.iter().any(|p| p.text == "123-45-6789"));
    }

    #[test]
    fn credit_card_detected() {
        let pii = scan_patterns("Card: 4111-1111-1111-1111 on file.");
        assert!(pii.iter().any(|p| p.text == "4111-1111-1111-1111"));
    }

    #[test]
    fn email_detected() {
        let pii = scan_patterns("Contact me at bob@example.com please.");
        assert!(pii.iter().any(|p| p.pii_type == "CONTACT"));
    }

    #[test]
    fn iban_detected() {
        assert!(looks_like_id_number("DE89370400440532013000"));
    }

    #[test]
    fn common_word_not_id() {
        assert!(!looks_like_id_number("Chemistry"));
    }

    #[test]
    fn address_with_zip() {
        assert!(looks_like_address("1234 Elm Street, Springfield, IL 62704"));
    }

    #[test]
    fn redact_replaces_pii() {
        let entities = vec![PiiEntity {
            text: "123-45-6789".to_string(),
            pii_type: "ID_NUMBER".to_string(),
            start: 11,
            end: 22,
            risk_level: "CRITICAL".to_string(),
        }];
        let result = redact("My SSN is 123-45-6789.", &entities);
        assert!(result.contains("[ID_NUMBER_1]"));
        assert!(!result.contains("123-45-6789"));
    }

    #[test]
    fn pseudonymize_consistent() {
        let entities = vec![
            PiiEntity {
                text: "bob@example.com".to_string(),
                pii_type: "CONTACT".to_string(),
                start: 0,
                end: 15,
                risk_level: "HIGH".to_string(),
            },
            PiiEntity {
                text: "555-867-5309".to_string(),
                pii_type: "CONTACT".to_string(),
                start: 20,
                end: 32,
                risk_level: "HIGH".to_string(),
            },
        ];
        let (result, mapping) = pseudonymize("bob@example.com --- 555-867-5309", &entities);
        assert!(mapping.get("bob@example.com").unwrap().contains('@'));
        assert!(mapping.get("555-867-5309").unwrap().starts_with("555-000-"));
        assert!(!result.contains("bob@example.com"));
    }

    #[test]
    fn report_counts() {
        let entities = vec![
            PiiEntity {
                text: "John".to_string(),
                pii_type: "PERSON".to_string(),
                start: 0,
                end: 4,
                risk_level: "LOW".to_string(),
            },
            PiiEntity {
                text: "123-45-6789".to_string(),
                pii_type: "ID_NUMBER".to_string(),
                start: 10,
                end: 21,
                risk_level: "CRITICAL".to_string(),
            },
        ];
        let r = report(&entities);
        assert_eq!(r.person_count, 1);
        assert_eq!(r.id_number_count, 1);
        assert!(r.k_anonymity_risk.starts_with("CRITICAL"));
    }
}
