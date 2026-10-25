//! PII (personally identifiable information) detection and redaction.
//!
//! Two detection paths:
//! - [`classify_entity()`](crate::pii::classify_entity): classifies NER entities as PII (uses character offsets from `Entity`)
//! - [`scan_patterns()`](crate::pii::scan_patterns): regex-based pre-NER scan for structured PII (SSN, credit card, IBAN, email, phone, address)
//!
//! After detection, use [`redact()`](crate::pii::redact) or [`pseudonymize()`](crate::pii::pseudonymize) to produce sanitized text.
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

mod grouped_iban;

use crate::Entity;
use regex::Regex;
use std::collections::HashMap;

/// US SSN structure check: the SSA never issues area 000, 666 or 900-999,
/// group 00, or serial 0000. Expects `AAA-GG-SSSS`.
fn ssn_valid(s: &str) -> bool {
    let parts: Vec<&str> = s.split('-').collect();
    let [area, group, serial] = parts.as_slice() else {
        return false;
    };
    let (Ok(area), Ok(group), Ok(serial)) = (
        area.parse::<u32>(),
        group.parse::<u32>(),
        serial.parse::<u32>(),
    ) else {
        return false;
    };
    area != 0 && area != 666 && area < 900 && group != 0 && serial != 0
}

/// Luhn check digit over the ASCII digits of `s` (separators ignored).
fn luhn_valid(s: &str) -> bool {
    let digits: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() < 12 {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == 1 {
                let d2 = d * 2;
                if d2 > 9 {
                    d2 - 9
                } else {
                    d2
                }
            } else {
                d
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

/// ISO 13616 IBAN check: 15 to 34 alphanumerics (whitespace and hyphens
/// ignored) whose rearranged numeric form is 1 mod 97.
pub(crate) fn iban_mod97_valid(raw: &str) -> bool {
    let s: String = raw
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .collect();
    if s.len() < 15 || s.len() > 34 {
        return false;
    }
    let s = s.to_ascii_uppercase();
    if !s.chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    let (head, tail) = s.split_at(4);
    let mut r: u32 = 0;
    for c in tail.chars().chain(head.chars()) {
        // Letters expand to two digits (A = 10 .. Z = 35).
        let v = c.to_digit(36).unwrap_or(0);
        r = if v >= 10 {
            (r * 100 + v) % 97
        } else {
            (r * 10 + v) % 97
        };
    }
    r == 1
}

const SSN_PATTERN: &str = r"\b\d{3}-\d{2}-\d{4}\b";
const CARD_PATTERN: &str = r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b";
const IBAN_PATTERN: &str = r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]{0,16})?\b";

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
/// This is independent of NER -- it catches structured PII via patterns.
/// Printed IBANs use uppercase groups separated by a single ASCII space or
/// hyphen at a registered country length. Group widths follow registry print
/// formats and may exceed four (BI, LY, SV). A separator after that endpoint
/// starts separate text, even if it is numeric.
/// Length and checksum checks do not establish account existence or BBAN validity.
/// Offsets are character offsets (Unicode scalar values), consistent with [`classify_entity`].
pub fn scan_patterns(text: &str) -> Vec<PiiEntity> {
    // Recognize complete grouped IBANs before shorter numeric patterns can
    // claim a portion of the account number.
    let mut results: Vec<PiiEntity> = grouped_iban::candidates(text)
        .map(|span| PiiEntity {
            text: text[span.clone()].to_string(),
            pii_type: "ID_NUMBER".to_string(),
            start: text[..span.start].chars().count(),
            end: text[..span.end].chars().count(),
            risk_level: "CRITICAL".to_string(),
        })
        .collect();

    type Check = Option<fn(&str) -> bool>;
    let patterns: &[(&str, &str, &str, Check)] = &[
        (SSN_PATTERN, "ID_NUMBER", "CRITICAL", Some(ssn_valid)),
        (CARD_PATTERN, "ID_NUMBER", "CRITICAL", Some(luhn_valid)),
        (
            IBAN_PATTERN,
            "ID_NUMBER",
            "CRITICAL",
            Some(iban_mod97_valid),
        ),
        (
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "CONTACT",
            "HIGH",
            None,
        ),
        (
            r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "CONTACT",
            "HIGH",
            None,
        ),
        (
            r"\b\d{1,5}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Circle|Cir|Terrace|Ter)\.?(?:,\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?\b",
            "ADDRESS",
            "HIGH",
            None,
        ),
    ];

    for &(pat, pii_type, risk, check) in patterns {
        if let Ok(re) = Regex::new(pat) {
            for m in re.find_iter(text) {
                if check.is_some_and(|valid| !valid(m.as_str())) {
                    continue;
                }
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
/// Entity offsets are character offsets (Unicode scalar values). Exact
/// duplicates are dropped and overlapping spans are merged into their union
/// before replacement: dropping an overlap instead would leave the part of the
/// longer span that extended past the shorter one in the output, which is a PII
/// leak. Back-to-front replacement keeps character offsets valid.
pub fn redact(text: &str, entities: &[PiiEntity]) -> String {
    let mut result = text.to_string();
    let mut type_counts: HashMap<&str, usize> = HashMap::new();

    // Deduplicate exact duplicates, then merge overlapping spans into their
    // union so every covered character is replaced. Sort by start ascending,
    // longest span first for ties.
    let mut sorted: Vec<&PiiEntity> = entities.iter().collect();
    sorted.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    sorted.dedup_by(|a, b| a.start == b.start && a.end == b.end);
    let mut merged: Vec<(usize, usize, &str)> = Vec::with_capacity(sorted.len());
    for entity in sorted {
        match merged.last_mut() {
            Some((_, end, _)) if entity.start < *end => {
                if entity.end > *end {
                    *end = entity.end;
                }
            }
            _ => merged.push((entity.start, entity.end, entity.pii_type.as_str())),
        }
    }
    // Reverse for back-to-front replacement (so char offsets stay valid).
    merged.reverse();

    for (start, end, pii_type) in merged {
        let count = type_counts.entry(pii_type).or_insert(0);
        *count += 1;
        let replacement = format!("[{}_{}]", pii_type, count);
        // Convert char offsets to byte offsets for replace_range
        let byte_start: usize = result.chars().take(start).map(|c| c.len_utf8()).sum();
        let byte_end: usize = result.chars().take(end).map(|c| c.len_utf8()).sum();
        result.replace_range(byte_start..byte_end, &replacement);
    }

    result
}

/// Redact structured PII (SSN, credit card, IBAN, email, phone, address) from
/// a string without loading an NER model.
///
/// Runs [`scan_patterns`] followed by [`redact`]. Suitable for log pipelines
/// and other hot paths where model load time is unacceptable; catches all
/// pattern-detectable PII but misses names (those require NER). For the full
/// pipeline including name detection, use [`scan_and_redact`].
///
/// # Example
///
/// ```
/// use anno::pii;
///
/// let scrubbed = pii::redact_patterns("SSN 123-45-6789 and email a@b.com");
/// assert!(scrubbed.contains("[ID_NUMBER_1]"));
/// assert!(scrubbed.contains("[CONTACT_1]"));
/// ```
pub fn redact_patterns(text: &str) -> String {
    let entities = scan_patterns(text);
    redact(text, &entities)
}

/// Replace each PII span with a fixed character (e.g. `'*'`), preserving length.
///
/// Useful for log display where position matters but content must be hidden.
/// Counts are character-level, not byte-level — `mask("héllo", ..., '*')` on the
/// entire span returns `"*****"` (5 chars), not `"******"`.
///
/// Entity offsets are character offsets. Overlapping spans are deduped the same
/// way [`redact`] deduplicates them.
pub fn mask(text: &str, entities: &[PiiEntity], fill: char) -> String {
    apply_per_entity(text, entities, |entity| {
        let width = entity.end.saturating_sub(entity.start);
        std::iter::repeat_n(fill, width).collect::<String>()
    })
}

/// Replace each PII span with a short fingerprint derived from the entity text.
///
/// The fingerprint is a 64-bit FxHash, hex-encoded. Same input always yields the
/// same fingerprint in the same process, which lets downstream systems correlate
/// occurrences of the same PII value without knowing its content. This is not
/// cryptographically secure — use it for log-scrub and analytics, not secrets.
///
/// Format: `[<TYPE>_<8-hex>]`, e.g. `"[PERSON_a1b2c3d4]"`.
pub fn fingerprint(text: &str, entities: &[PiiEntity]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    apply_per_entity(text, entities, |entity| {
        let mut h = DefaultHasher::new();
        entity.text.hash(&mut h);
        format!(
            "[{}_{:08x}]",
            entity.pii_type,
            (h.finish() & 0xFFFF_FFFF) as u32
        )
    })
}

/// Apply a caller-supplied replacement function to each PII span.
///
/// Generic version of [`redact`] / [`mask`] / [`fingerprint`] — use this when
/// the built-in operators don't fit. `replacement_fn` is called once per
/// entity after dedup+sort; the return value replaces that span.
///
/// Entity offsets are character offsets; internal byte-offset conversion is
/// handled here.
pub fn replace<F>(text: &str, entities: &[PiiEntity], mut replacement_fn: F) -> String
where
    F: FnMut(&PiiEntity) -> String,
{
    apply_per_entity(text, entities, |e| replacement_fn(e))
}

/// Shared core for `redact` / `mask` / `fingerprint` / `replace`.
///
/// Deduplicates overlapping spans greedily (keeps longest-at-start), then
/// walks them back-to-front so earlier char offsets remain valid during
/// `replace_range` calls.
fn apply_per_entity<F>(text: &str, entities: &[PiiEntity], mut replacement_fn: F) -> String
where
    F: FnMut(&PiiEntity) -> String,
{
    let mut result = text.to_string();

    let mut sorted: Vec<_> = entities.iter().collect();
    sorted.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    sorted.dedup_by(|a, b| a.start == b.start && a.end == b.end);
    let mut max_end = 0;
    sorted.retain(|e| {
        if e.start < max_end {
            false
        } else {
            max_end = e.end;
            true
        }
    });
    sorted.reverse();

    for entity in sorted {
        let byte_start: usize = result
            .chars()
            .take(entity.start)
            .map(|c| c.len_utf8())
            .sum();
        let byte_end: usize = result.chars().take(entity.end).map(|c| c.len_utf8()).sum();
        let replacement = replacement_fn(entity);
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
    sorted.sort_by_key(|b| std::cmp::Reverse(b.start));

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
    dedup_overlapping(&mut pii_entities);
    Ok(redact(text, &pii_entities))
}

/// Remove duplicate and overlapping PII entities, keeping the longest span.
///
/// After merging NER-based and regex-based detections, duplicates and overlaps
/// are common (e.g., NER finds "John Smith" and regex finds "123-45-6789" within
/// a span the NER also matched). This function sorts by start offset, then
/// greedily keeps the longest non-overlapping spans.
fn dedup_overlapping(entities: &mut Vec<PiiEntity>) {
    // Sort by start, then longest span first for ties
    entities.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    // Dedup exact duplicates
    entities.dedup_by(|a, b| a.start == b.start && a.end == b.end);
    // Merge overlaps into their union: dropping a span that overlaps a kept one
    // would remove it from the reported entities and leave its uncovered tail
    // unredacted. The merged span keeps the type of the earliest-starting span.
    let mut merged: Vec<PiiEntity> = Vec::with_capacity(entities.len());
    for entity in entities.drain(..) {
        match merged.last_mut() {
            Some(last) if entity.start < last.end => {
                if entity.end > last.end {
                    last.end = entity.end;
                }
            }
            _ => merged.push(entity),
        }
    }
    *entities = merged;
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
        "St",
        "Street",
        "Ave",
        "Avenue",
        "Rd",
        "Road",
        "Blvd",
        "Boulevard",
        "Dr",
        "Drive",
        "Lane",
        "Ln",
        "Way",
        "Court",
        "Ct",
        "Place",
        "Pl",
        "Circle",
        "Cir",
        "Terrace",
        "Ter",
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
    if grouped_iban::candidates(text).next().is_some() {
        return true;
    }
    // The heuristic extractor accepts a registered IBAN with a single space
    // at any position ("DE89 370400440532013000"), which the grouped layout
    // above rejects. Fall back to the checksum so classification agrees with
    // extraction instead of letting a shorter numeric pattern claim the tail.
    if iban_mod97_valid(text) {
        return true;
    }
    type Checked = (&'static str, fn(&str) -> bool);
    let checked: [Checked; 3] = [
        (r"\d{3}-\d{2}-\d{4}", ssn_valid),
        (r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}", luhn_valid),
        (
            r"[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]{0,16})?",
            iban_mod97_valid,
        ),
    ];
    for (pat, valid) in checked {
        if let Ok(re) = Regex::new(pat) {
            if re.find_iter(text).any(|m| valid(m.as_str())) {
                return true;
            }
        }
    }
    // Alphanumeric catch-all for short ID-like tokens (e.g. MRNs, short codes).
    // Require that digits make up at least half the characters to avoid
    // false-positives on version strings like "Python3", "iPhone6", "Cent0S".
    let digit_count = text.chars().filter(|c| c.is_ascii_digit()).count();
    if text.len() >= 6
        && text.len() <= 10
        && text.chars().all(|c| c.is_alphanumeric())
        && digit_count * 2 >= text.len()
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EntityCategory, EntityType};

    /// Character offsets of the first occurrence of `needle` in `text`.
    fn char_span_of(text: &str, needle: &str) -> (usize, usize) {
        let byte_start = text
            .find(needle)
            .unwrap_or_else(|| panic!("{needle:?} not found in {text:?}"));
        let start = text[..byte_start].chars().count();
        (start, start + needle.chars().count())
    }

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
    fn version_strings_not_id() {
        assert!(!looks_like_id_number("Python3"));
        assert!(!looks_like_id_number("Win10"));
        assert!(!looks_like_id_number("iPhone6"));
        assert!(!looks_like_id_number("Cent0S"));
    }

    #[test]
    fn address_with_zip() {
        assert!(looks_like_address("1234 Elm Street, Springfield, IL 62704"));
    }

    #[test]
    fn redact_replaces_pii() {
        // "My SSN is " = 10 chars, "123-45-6789" = chars 10..21
        let entities = vec![PiiEntity {
            text: "123-45-6789".to_string(),
            pii_type: "ID_NUMBER".to_string(),
            start: 10,
            end: 21,
            risk_level: "CRITICAL".to_string(),
        }];
        let result = redact("My SSN is 123-45-6789.", &entities);
        assert_eq!(result, "My SSN is [ID_NUMBER_1].");
    }

    #[test]
    fn redact_non_ascii() {
        // "caf\u{e9}" is 4 chars (e with accent = 1 char, 2 bytes)
        let text = "caf\u{e9} SSN: 123-45-6789.";
        let entities = vec![PiiEntity {
            text: "123-45-6789".to_string(),
            pii_type: "ID_NUMBER".to_string(),
            start: 10, // "caf\u{e9} SSN: " = 10 chars
            end: 21,   // 10 + 11 chars
            risk_level: "CRITICAL".to_string(),
        }];
        let result = redact(text, &entities);
        assert_eq!(result, "caf\u{e9} SSN: [ID_NUMBER_1].");
        assert!(!result.contains("123-45-6789"));
    }

    #[test]
    fn scan_patterns_returns_char_offsets() {
        let text = "caf\u{e9} SSN: 123-45-6789 end";
        let pii = scan_patterns(text);
        let ssn = pii.iter().find(|p| p.text == "123-45-6789");
        assert!(ssn.is_some(), "should detect SSN");
        let ssn = ssn.unwrap();
        // Verify these are char offsets, not byte offsets
        let extracted: String = text
            .chars()
            .skip(ssn.start)
            .take(ssn.end - ssn.start)
            .collect();
        assert_eq!(extracted, "123-45-6789");
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
    fn pseudonymize_same_entity_gets_same_pseudonym() {
        // The same entity text appearing twice should produce the same pseudonym.
        let entities = vec![
            PiiEntity {
                text: "John Smith".to_string(),
                pii_type: "PERSON".to_string(),
                start: 0,
                end: 10,
                risk_level: "MEDIUM".to_string(),
            },
            PiiEntity {
                text: "John Smith".to_string(),
                pii_type: "PERSON".to_string(),
                start: 15,
                end: 25,
                risk_level: "MEDIUM".to_string(),
            },
        ];
        let text = "John Smith met John Smith again.";
        let (result, mapping) = pseudonymize(text, &entities);
        let fake = mapping
            .get("John Smith")
            .expect("mapping should contain John Smith");
        // Both occurrences should be replaced with the same pseudonym
        assert_eq!(
            result.matches(fake.as_str()).count(),
            2,
            "Both occurrences of 'John Smith' should map to the same pseudonym '{}', got: {}",
            fake,
            result
        );
    }

    #[test]
    fn redact_overlapping_spans_no_panic() {
        // Overlapping spans should be resolved gracefully (no panic, no garbled output).
        // The implementation drops the inner span, keeping the outer one.
        let entities = vec![
            PiiEntity {
                text: "John Smith".to_string(),
                pii_type: "PERSON".to_string(),
                start: 0,
                end: 10,
                risk_level: "MEDIUM".to_string(),
            },
            PiiEntity {
                // Overlaps with "John Smith"
                text: "John".to_string(),
                pii_type: "PERSON".to_string(),
                start: 0,
                end: 4,
                risk_level: "LOW".to_string(),
            },
        ];
        let text = "John Smith called.";
        let result = redact(text, &entities);
        // Assert the exact output, not just the absence of the original string. A resolver
        // that kept the inner span would emit "[PERSON_1] Smith called.", which still
        // satisfies a "does not contain 'John Smith'" check while leaking "Smith".
        assert_eq!(result, "[PERSON_1] called.");
        assert!(
            !result.contains("Smith"),
            "no part of the overlapping span may survive redaction"
        );

        // The resolver sorts spans itself, so the input order must not change the result.
        let mut reversed = entities.clone();
        reversed.reverse();
        assert_eq!(redact(text, &reversed), "[PERSON_1] called.");
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

    #[test]
    fn mask_preserves_length_and_position() {
        let text = "John met Alice.";
        let entities = vec![
            PiiEntity {
                text: "John".to_string(),
                pii_type: "PERSON".to_string(),
                start: 0,
                end: 4,
                risk_level: "LOW".to_string(),
            },
            PiiEntity {
                text: "Alice".to_string(),
                pii_type: "PERSON".to_string(),
                start: 9,
                end: 14,
                risk_level: "LOW".to_string(),
            },
        ];
        let masked = mask(text, &entities, '*');
        assert_eq!(masked, "**** met *****.");
    }

    #[test]
    fn mask_handles_multibyte_unicode() {
        // "café" is 4 chars (4 code points) but 5 bytes in UTF-8.
        // mask works in character-space, so the result has 4 fill chars.
        let text = "café alice";
        let entities = vec![PiiEntity {
            text: "café".to_string(),
            pii_type: "PERSON".to_string(),
            start: 0,
            end: 4,
            risk_level: "LOW".to_string(),
        }];
        let masked = mask(text, &entities, '#');
        assert_eq!(masked, "#### alice");
    }

    #[test]
    fn fingerprint_is_deterministic_same_input() {
        let text = "John met John.";
        let entities = vec![
            PiiEntity {
                text: "John".to_string(),
                pii_type: "PERSON".to_string(),
                start: 0,
                end: 4,
                risk_level: "LOW".to_string(),
            },
            PiiEntity {
                text: "John".to_string(),
                pii_type: "PERSON".to_string(),
                start: 9,
                end: 13,
                risk_level: "LOW".to_string(),
            },
        ];
        let fp = fingerprint(text, &entities);
        // Both occurrences of "John" should receive the same fingerprint.
        let tokens: Vec<&str> = fp
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '[' && c != ']')
            .filter(|s| s.starts_with("[PERSON_") && s.ends_with(']'))
            .collect();
        assert_eq!(
            tokens.len(),
            2,
            "expected two fingerprint tokens, got {fp:?}"
        );
        assert_eq!(
            tokens[0], tokens[1],
            "identical entity text must produce identical fingerprint"
        );
    }

    #[test]
    fn replace_applies_caller_fn() {
        let text = "SSN 123-45-6789 recorded.";
        let entities = vec![PiiEntity {
            text: "123-45-6789".to_string(),
            pii_type: "ID_NUMBER".to_string(),
            start: 4,
            end: 15,
            risk_level: "CRITICAL".to_string(),
        }];
        let replaced = replace(text, &entities, |e| format!("<{}>", e.pii_type));
        assert_eq!(replaced, "SSN <ID_NUMBER> recorded.");
    }

    fn ids(text: &str) -> Vec<String> {
        scan_patterns(text)
            .into_iter()
            .filter(|p| p.pii_type == "ID_NUMBER")
            .map(|p| p.text)
            .collect()
    }

    /// SSA never issues area 000, 666 or 900-999, group 00, or serial 0000.
    #[test]
    fn scan_rejects_ssns_with_unissued_area_group_or_serial() {
        for bad in [
            "000-12-3456",
            "666-12-3456",
            "912-12-3456",
            "123-00-4567",
            "123-45-0000",
        ] {
            assert!(ids(&format!("SSN {bad} here")).is_empty(), "{bad} accepted");
            assert!(!looks_like_id_number(bad), "{bad} accepted");
        }
        assert_eq!(ids("SSN 123-45-6789 here"), vec!["123-45-6789"]);
    }

    /// Card numbers must pass the Luhn check digit.
    #[test]
    fn scan_requires_luhn_valid_card_numbers() {
        assert!(ids("Card 4111 1111 1111 1112 on file").is_empty());
        assert!(!looks_like_id_number("4111-1111-1111-1112"));
        assert_eq!(
            ids("Card 4111 1111 1111 1111 on file"),
            vec!["4111 1111 1111 1111"]
        );
        assert_eq!(
            ids("Card 5500005555555559 on file"),
            vec!["5500005555555559"]
        );
    }

    /// IBANs must pass the ISO 13616 mod-97 check, as in the French backend.
    #[test]
    fn scan_requires_mod97_valid_ibans() {
        assert!(ids("IBAN DE89370400440532013001 x").is_empty());
        assert!(!looks_like_id_number("DE89370400440532013001"));
        assert_eq!(
            ids("IBAN DE89370400440532013000 x"),
            vec!["DE89370400440532013000"]
        );
        assert_eq!(
            ids("IBAN GB82WEST12345698765432 x"),
            vec!["GB82WEST12345698765432"]
        );
    }

    #[test]
    fn grouped_iban_public_spans_and_redaction() {
        for token in [
            "DE89 3704 0044 0532 0130 00",
            "GB82 WEST 1234 5698 7654 32",
            "FR14 2004 1010 0505 0001 3M02 606",
        ] {
            for suffix in [".", " SEND NOW.", " 1234"] {
                let text = format!("☎ café e\u{301} {token}{suffix}");
                let entities = scan_patterns(&text);
                assert_eq!(entities.len(), 1, "{text}: {entities:?}");
                let entity = &entities[0];
                assert_eq!(entity.text, token);
                assert_eq!(entity.pii_type, "ID_NUMBER");
                assert_eq!((entity.start, entity.end), (10, 10 + token.len()));
                assert_eq!(
                    text.chars()
                        .skip(entity.start)
                        .take(entity.end - entity.start)
                        .collect::<String>(),
                    token
                );
                assert!(looks_like_id_number(token));
                assert_eq!(
                    redact_patterns(&text),
                    format!("☎ café e\u{301} [ID_NUMBER_1]{suffix}")
                );
            }
        }
    }

    #[test]
    fn grouped_iban_rejects_malformed_candidates() {
        for token in [
            "DE89 3704 0044 0532 0130 01",  // wrong checksum
            "ZZ66 0000 0000 0000 0000 00",  // checksum-valid unknown country
            "DE89  3704 0044 0532 0130 00", // double space
            "DE89\t3704 0044 0532 0130 00",
            "DE89\n3704 0044 0532 0130 00",
            "DE89\u{a0}3704 0044 0532 0130 00",
            "DE89 370 0044 0532 0130 00",   // short interior group
            "DE89 3704 0044 0532 0130 0",   // short final group
            "DE89 3704 0044 0532 0130 000", // long final group
            "DE89 3704 0044 0532 0130 00X",
            "DE89 3704 0044 0532 0130 00_",
            "DE89 3704 0044 0532 0130 00\u{301}",
            "DE89 3704 0044 0532 0130 00α",
            "αDE89 3704 0044 0532 0130 00",
            "_DE89 3704 0044 0532 0130 00",
            "de89 3704 0044 0532 0130 00",
            "GB82 west 1234 5698 7654 32",
        ] {
            assert!(grouped_iban::candidates(token).next().is_none(), "{token}");
            assert!(
                scan_patterns(token)
                    .iter()
                    .all(|entity| entity.text != token),
                "malformed grouped token was classified whole: {token}"
            );
            // A rejected grouped candidate may still contain independently
            // recognized PII (for example, a Luhn-valid run of zeroes).
        }
    }

    #[test]
    fn grouped_iban_preserves_compact_acceptance() {
        for token in [
            "DE89370400440532013000",
            "GB82WEST12345698765432",
            "ZZ66000000000000000000",
        ] {
            let text = format!("{token} SEND NOW.");
            assert_eq!(ids(&text), vec![token]);
            assert!(looks_like_id_number(token));
            assert_eq!(redact_patterns(&text), "[ID_NUMBER_1] SEND NOW.");
        }
    }

    #[cfg(feature = "heuristic-fr")]
    #[test]
    fn grouped_iban_ner_classification_and_redaction() {
        use crate::backends::heuristic_fr::{iban_intl::extract_iban_intl, HeuristicFrNer};
        let text = "café DE89 3704 0044 0532 0130 00.";
        let entities = extract_iban_intl(text);
        assert_eq!(entities.len(), 1);
        let classified = classify_entity(&entities[0]).expect("extracted IBAN is PII");
        assert_eq!((classified.start, classified.end), (5, 32));
        assert_eq!(classified.text, "DE89 3704 0044 0532 0130 00");
        for suffix in [".", " SEND NOW.", " 1234"] {
            let text = format!("café DE89 3704 0044 0532 0130 00{suffix}");
            assert_eq!(
                scan_and_redact(&text, &HeuristicFrNer::new()).unwrap(),
                format!("café [ID_NUMBER_1]{suffix}")
            );
        }
    }

    /// Registry print formats group by account component, not in fours. The
    /// whole registered span must be redacted, not just a numeric tail.
    #[test]
    fn grouped_iban_registry_component_groupings() {
        for token in [
            "BI13 20001 10001 00001234567 89",
            "LY83 002 048 000020100120361",
            "SV62 CENR 0000 0000 0000 0070 0025",
            "SV43 ACAT 00000000000000123123",
        ] {
            let text = format!("cafe {token} END");
            let entities = scan_patterns(&text);
            assert_eq!(entities.len(), 1, "{text}: {entities:?}");
            assert_eq!(entities[0].text, token);
            assert_eq!((entities[0].start, entities[0].end), (5, 5 + token.len()));
            assert!(looks_like_id_number(token));
            assert_eq!(ids(token), vec![token]);
            assert_eq!(redact_patterns(&text), "cafe [ID_NUMBER_1] END");
        }
    }

    /// AT's 16-digit BBAN run is Luhn-valid, so the card pattern used to claim
    /// it and leave the `AT61 ` prefix in cleartext. It is one IBAN span.
    #[test]
    fn grouped_iban_at_luhn_inner_run_is_one_span() {
        let token = "AT61 1904 3002 3457 3201";
        let text = format!("{token} SEND NOW.");
        assert_eq!(ids(&text), vec![token]);
        assert_eq!(redact_patterns(&text), "[ID_NUMBER_1] SEND NOW.");
    }

    /// Country lengths divisible by four exercise the four-group fast path.
    #[test]
    fn grouped_iban_lengths_divisible_by_four() {
        for token in ["AT61 1904 3002 3457 3201", "BE68 5390 0754 7034"] {
            let text = format!("x {token} y");
            let entities = scan_patterns(&text);
            assert_eq!(entities.len(), 1, "{text}: {entities:?}");
            assert_eq!(entities[0].text, token);
            assert_eq!((entities[0].start, entities[0].end), (2, 2 + token.len()));
            assert!(looks_like_id_number(token));
            assert_eq!(redact_patterns(&text), "x [ID_NUMBER_1] y");
        }
    }

    /// Hyphens are printed separators, accepted exactly like single spaces.
    #[test]
    fn grouped_iban_hyphenated_form() {
        let token = "DE89-3704-0044-0532-0130-00";
        assert!(looks_like_id_number(token));
        assert_eq!(ids(token), vec![token]);
        assert_eq!(redact_patterns(token), "[ID_NUMBER_1]");
        assert_eq!(
            redact_patterns(&format!("{token} SEND")),
            "[ID_NUMBER_1] SEND"
        );
        let text = format!("cafe {token} END");
        assert_eq!(redact_patterns(&text), "cafe [ID_NUMBER_1] END");
    }

    /// Two complete grouped IBANs separated by one space are distinct spans.
    #[test]
    fn grouped_iban_adjacent_tokens() {
        let text = "DE89 3704 0044 0532 0130 00 GB82 WEST 1234 5698 7654 32";
        let entities = scan_patterns(text);
        assert_eq!(entities.len(), 2, "{entities:?}");
        assert_eq!(entities[0].text, "DE89 3704 0044 0532 0130 00");
        assert_eq!((entities[0].start, entities[0].end), (0, 27));
        assert_eq!(entities[1].text, "GB82 WEST 1234 5698 7654 32");
        assert_eq!((entities[1].start, entities[1].end), (28, 55));
        let redacted = redact_patterns(text);
        assert_eq!(redacted.matches("[ID_NUMBER_").count(), 2);
        assert!(!redacted.contains("DE89"));
        assert!(!redacted.contains("GB82"));
    }

    /// `mask` preserves character count, including the grouping separators.
    #[test]
    fn grouped_iban_mask_preserves_span_length() {
        for token in [
            "DE89 3704 0044 0532 0130 00",
            "BI13 20001 10001 00001234567 89",
        ] {
            let text = format!("x {token} y");
            let masked = mask(&text, &scan_patterns(&text), '*');
            assert_eq!(masked, format!("x {} y", "*".repeat(token.len())));
            assert_eq!(masked.chars().count(), text.chars().count());
        }
    }

    /// Lower-case grouped prints stay outside the grouped pattern route (the
    /// documented gap). The checksum fallback still classifies the token when
    /// a caller passes it directly to `looks_like_id_number`.
    #[test]
    fn grouped_iban_lowercase_form_remains_ungrouped() {
        let token = "de89 3704 0044 0532 0130 00";
        assert!(grouped_iban::candidates(token).next().is_none());
        assert!(scan_patterns(token)
            .iter()
            .all(|entity| entity.text != token));
        assert_eq!(redact_patterns(token), token);
        assert!(looks_like_id_number(token));
    }

    /// The heuristic extractor accepts a single space at any position inside a
    /// registered IBAN. Classification must agree, otherwise the bank-account
    /// tail is handed to the phone pattern and survives redaction.
    #[cfg(feature = "heuristic-fr")]
    #[test]
    fn grouped_iban_extractor_classifier_agree_on_split_layout() {
        use crate::backends::heuristic_fr::{iban_intl::extract_iban_intl, HeuristicFrNer};
        let token = "DE89 370400440532013000";
        assert!(looks_like_id_number(token));
        let text = format!("cafe {token}");
        let entities = extract_iban_intl(&text);
        assert_eq!(entities.len(), 1, "{entities:?}");
        let classified = classify_entity(&entities[0]).expect("split IBAN is PII");
        assert_eq!(classified.text, token);
        assert_eq!(classified.pii_type, "ID_NUMBER");
        assert_eq!(
            scan_and_redact(&text, &HeuristicFrNer::new()).unwrap(),
            "cafe [ID_NUMBER_1]"
        );
    }

    /// PHONE is wired into `scan_patterns` but every existing test exercised a
    /// different pattern. Assert the whole returned span for each US/NANP form
    /// the pattern claims, then assert the invalid shapes produce no match.
    #[test]
    fn scan_patterns_phone_exact_spans() {
        let valid = [
            ("call +1 555-867-5309 now", "+1 555-867-5309"),
            ("call +1 (555) 867-5309 now", "+1 (555) 867-5309"),
            ("call (555) 867-5309 now", "(555) 867-5309"),
            ("call 555.867.5309 now", "555.867.5309"),
            ("call 555 867 5309 now", "555 867 5309"),
            ("call 5558675309 now", "5558675309"),
        ];
        for (text, expected) in valid {
            let entities = scan_patterns(text);
            let phones: Vec<&PiiEntity> = entities
                .iter()
                .filter(|e| e.pii_type == "CONTACT")
                .collect();
            assert_eq!(phones.len(), 1, "{text}: {entities:?}");
            assert_eq!(phones[0].text, expected, "{text}");
            assert_eq!(
                (phones[0].start, phones[0].end),
                char_span_of(text, expected),
                "{text}"
            );
        }

        for text in [
            "call 555-8675 now",         // too short
            "call ABC-DEF-GHIJ now",     // letters
            "call +44 20 7946 0958 now", // non-NANP prefix is not claimed
        ] {
            let entities = scan_patterns(text);
            assert!(
                entities.iter().all(|e| e.pii_type != "CONTACT"),
                "{text}: {entities:?}"
            );
        }
    }

    /// ADDRESS requires a street number plus a capitalized street name and a
    /// recognized suffix. Number-only and street-name-only text must not match.
    #[test]
    fn scan_patterns_address_exact_spans() {
        let valid = [
            ("at 742 Evergreen Terrace today", "742 Evergreen Terrace"),
            (
                "at 1234 Elm Street, Springfield, IL 62704 today",
                "1234 Elm Street, Springfield, IL 62704",
            ),
            ("1234 Elm Street", "1234 Elm Street"),
        ];
        for (text, expected) in valid {
            let entities = scan_patterns(text);
            let addresses: Vec<&PiiEntity> = entities
                .iter()
                .filter(|e| e.pii_type == "ADDRESS")
                .collect();
            assert_eq!(addresses.len(), 1, "{text}: {entities:?}");
            assert_eq!(addresses[0].text, expected, "{text}");
            assert_eq!(
                (addresses[0].start, addresses[0].end),
                char_span_of(text, expected),
                "{text}"
            );
        }

        for text in [
            "at 1234 today",
            "at Elm Street today",
            "at 1234 elm street today",
            "at 1234 Elm today",
            "The White House",
        ] {
            let entities = scan_patterns(text);
            assert!(
                entities.iter().all(|e| e.pii_type != "ADDRESS"),
                "{text}: {entities:?}"
            );
        }
    }

    /// `looks_like_address` is the NER location gate. It previously had only a
    /// positive test, so a rule that accepted any text containing a digit or a
    /// state abbreviation would have gone unnoticed.
    #[test]
    fn looks_like_address_requires_number_and_street() {
        for text in ["1234", "Elm Street", "Main Street", "Springfield"] {
            assert!(!looks_like_address(text), "{text}");
        }
        for text in ["1234 Elm Street", "1234 Elm Street, Springfield, IL 62704"] {
            assert!(looks_like_address(text), "{text}");
        }
    }

    /// The scanner's ADDRESS pattern accepts "Terrace", but the NER location
    /// gate's street list does not. Text that `scan_patterns` reports as one
    /// ADDRESS span must also classify, or `classify_entity` silently drops an
    /// address that the pattern path redacts.
    #[test]
    fn looks_like_address_agrees_with_scanner_on_terrace() {
        let text = "at 742 Evergreen Terrace today";
        let entities = scan_patterns(text);
        let addresses: Vec<&PiiEntity> = entities
            .iter()
            .filter(|e| e.pii_type == "ADDRESS")
            .collect();
        assert_eq!(addresses.len(), 1, "scanner must report the address");
        assert!(
            looks_like_address("742 Evergreen Terrace"),
            "classifier street list disagrees with the scanner's Terrace suffix"
        );
    }

    /// NER-to-PII mapping for a person. Offsets must be the entity's character
    /// offsets, not a re-scan of the text.
    #[test]
    fn classify_person_entity() {
        let entity = Entity::new("John Smith", EntityType::Person, 4, 14, 0.95);
        let pii = classify_entity(&entity).expect("person entity is PII");
        assert_eq!(pii.pii_type, "PERSON");
        assert_eq!(pii.text, "John Smith");
        assert_eq!((pii.start, pii.end), (4, 14));
        assert_eq!(pii.risk_level, "MEDIUM");
    }

    /// Email- and phone-valued entities map to CONTACT rather than falling
    /// through to the ID-number heuristic.
    #[test]
    fn classify_contact_entities() {
        let email_text = "bob@example.com";
        let email = Entity::new(
            email_text,
            EntityType::Email,
            10,
            10 + email_text.chars().count(),
            0.99,
        );
        let pii = classify_entity(&email).expect("email entity is PII");
        assert_eq!(pii.pii_type, "CONTACT");
        assert_eq!(pii.risk_level, "HIGH");
        assert_eq!((pii.start, pii.end), (10, 10 + email_text.chars().count()));

        let phone_text = "+1 555-867-5309";
        let phone = Entity::new(
            phone_text,
            EntityType::Phone,
            3,
            3 + phone_text.chars().count(),
            0.9,
        );
        let pii = classify_entity(&phone).expect("phone entity is PII");
        assert_eq!(pii.pii_type, "CONTACT");
        assert_eq!((pii.start, pii.end), (3, 3 + phone_text.chars().count()));
    }

    /// An IBAN-valued extractor entity maps to one ID_NUMBER span.
    #[test]
    fn classify_iban_entity() {
        for text in ["DE89370400440532013000", "GB82WEST12345698765432"] {
            let entity = Entity::new(
                text,
                EntityType::Custom {
                    name: "iban".into(),
                    category: EntityCategory::Numeric,
                },
                0,
                text.chars().count(),
                1.0,
            );
            let pii = classify_entity(&entity).expect("compact IBAN is PII");
            assert_eq!(pii.pii_type, "ID_NUMBER", "{text}");
            assert_eq!(pii.risk_level, "CRITICAL", "{text}");
            assert_eq!((pii.start, pii.end), (0, text.chars().count()), "{text}");
        }
    }

    /// The heuristic extractor accepts a registered IBAN split after the
    /// country code ("DE89 370400440532013000"). That text passes the mod-97
    /// checksum but is not the four-character print layout. It must classify as
    /// one ID_NUMBER, otherwise the numeric tail is handed to the phone pattern
    /// and survives redaction. The lowercase form can only be recognized by the
    /// checksum fallback, which pins that path independently of the grouped
    /// parser.
    #[test]
    fn classify_split_layout_iban_entity() {
        let split = "DE89 370400440532013000";
        assert!(iban_mod97_valid(split));
        let lowercase = "de89 370400440532013000";
        assert!(iban_mod97_valid(lowercase));
        assert!(grouped_iban::candidates(lowercase).next().is_none());

        for text in [split, lowercase] {
            let entity = Entity::new(
                text,
                EntityType::Custom {
                    name: "iban".into(),
                    category: EntityCategory::Numeric,
                },
                0,
                text.chars().count(),
                1.0,
            );
            let pii = classify_entity(&entity).expect("split IBAN is PII");
            assert_eq!(pii.pii_type, "ID_NUMBER", "{text}");
            assert_eq!((pii.start, pii.end), (0, text.chars().count()), "{text}");
        }
    }

    /// Non-PII NER entities return `None` instead of a low-confidence guess.
    #[test]
    fn classify_non_pii_entities_return_none() {
        let cases = [
            ("Acme Corp", EntityType::Organization),
            ("January 5", EntityType::Date),
            ("https://example.com", EntityType::Url),
            ("$500", EntityType::Money),
            ("Paris", EntityType::Location),
        ];
        for (text, entity_type) in cases {
            let entity = Entity::new(text, entity_type, 0, text.chars().count(), 0.9);
            assert!(classify_entity(&entity).is_none(), "{text}");
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Turn a span seed list into disjoint spans by walking a cursor. Every
    /// returned range is non-empty and inside `0..text_len`.
    fn disjoint_spans(text_len: usize, raw: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut cursor = 0;
        for &(gap_seed, len_seed) in raw {
            if cursor >= text_len {
                break;
            }
            let start = cursor + gap_seed % 5;
            if start >= text_len {
                break;
            }
            let end = (start + 1 + len_seed % 6).min(text_len);
            spans.push((start, end));
            cursor = end;
        }
        spans
    }

    /// Turn a span seed list into arbitrary, possibly overlapping spans.
    fn arbitrary_spans(text_len: usize, raw: &[(usize, usize)]) -> Vec<(usize, usize)> {
        if text_len == 0 {
            return Vec::new();
        }
        raw.iter()
            .map(|&(start_seed, len_seed)| {
                let start = start_seed % text_len;
                let end = (start + 1 + len_seed % 6).min(text_len);
                (start, end)
            })
            .collect()
    }

    fn entities_from_spans(text: &str, spans: &[(usize, usize)]) -> Vec<PiiEntity> {
        const TYPES: [&str; 3] = ["PERSON", "CONTACT", "ID_NUMBER"];
        spans
            .iter()
            .enumerate()
            .map(|(i, &(start, end))| PiiEntity {
                text: text.chars().skip(start).take(end - start).collect(),
                pii_type: TYPES[i % TYPES.len()].to_string(),
                start,
                end,
                risk_level: "HIGH".to_string(),
            })
            .collect()
    }

    fn covered(text_len: usize, spans: &[(usize, usize)]) -> Vec<bool> {
        let mut covered = vec![false; text_len];
        for &(start, end) in spans {
            for flag in covered.iter_mut().take(end).skip(start) {
                *flag = true;
            }
        }
        covered
    }

    /// Oracle: keep exactly the characters the spans do not cover, in order.
    fn survivors(text: &str, covered: &[bool]) -> String {
        text.chars()
            .zip(covered.iter().copied())
            .filter(|&(_, is_covered)| !is_covered)
            .map(|(ch, _)| ch)
            .collect()
    }

    fn is_placeholder(token: &str) -> bool {
        let Some((kind, number)) = token.rsplit_once('_') else {
            return false;
        };
        !kind.is_empty()
            && kind.chars().all(|c| c.is_ascii_uppercase() || c == '_')
            && !number.is_empty()
            && number.chars().all(|c| c.is_ascii_digit())
    }

    /// Remove the `[TYPE_n]` tokens `redact` inserts, leaving only source text.
    fn strip_placeholders(output: &str) -> String {
        let mut survivors = String::with_capacity(output.len());
        let mut rest = output;
        while let Some(open) = rest.find('[') {
            survivors.push_str(&rest[..open]);
            let after_open = &rest[open + 1..];
            match after_open.find(']') {
                Some(close) if is_placeholder(&after_open[..close]) => {
                    rest = &after_open[close + 1..];
                }
                _ => {
                    survivors.push('[');
                    rest = after_open;
                }
            }
        }
        survivors.push_str(rest);
        survivors
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Non-overlapping spans: every chosen range is replaced, and the text
        /// outside those ranges is preserved character-for-character. The
        /// expected survivors come from the span list, not from `redact`.
        #[test]
        fn redact_disjoint_spans_never_leaks_span_text(
            text in "[a-zA-Z0-9 .@_-]{0,48}",
            raw in proptest::collection::vec((0usize..8, 0usize..6), 0..=5),
        ) {
            let text_len = text.chars().count();
            let spans = disjoint_spans(text_len, &raw);
            let entities = entities_from_spans(&text, &spans);
            let output = redact(&text, &entities);

            let covered = covered(text_len, &spans);
            let expected_survivors = survivors(&text, &covered);
            let actual_survivors = strip_placeholders(&output);
            prop_assert_eq!(
                &actual_survivors,
                &expected_survivors,
                "text={:?} spans={:?} output={:?}",
                text,
                spans,
                output
            );

            // The generated alphabet has no brackets, so every '[' is a token.
            prop_assert_eq!(
                output.matches('[').count(),
                spans.len(),
                "text={:?} spans={:?} output={:?}",
                text,
                spans,
                output
            );
        }

        /// Overlapping spans: `redact` must not panic, and no character in the
        /// union of the chosen spans may survive. A resolver that keeps only the
        /// first span of a partial overlap leaves the longer span's tail in the
        /// output and fails this property.
        #[test]
        fn redact_overlapping_spans_remove_union_without_panic(
            text in "[a-zA-Z0-9 .@_-]{0,48}",
            raw in proptest::collection::vec((0usize..48, 0usize..6), 1..=5),
        ) {
            let text_len = text.chars().count();
            if text_len == 0 {
                return Ok(());
            }
            let spans = arbitrary_spans(text_len, &raw);
            let entities = entities_from_spans(&text, &spans);

            let output = redact(&text, &entities);

            let covered = covered(text_len, &spans);
            let expected_survivors = survivors(&text, &covered);
            let actual_survivors = strip_placeholders(&output);
            prop_assert_eq!(
                &actual_survivors,
                &expected_survivors,
                "text={:?} spans={:?} output={:?}",
                text,
                spans,
                output
            );
        }
    }
}
