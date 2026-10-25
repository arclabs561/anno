//! Candidate boundaries for printed IBANs; compact recognition remains separate.

use regex::Regex;
use std::{ops::Range, sync::LazyLock};

#[path = "iban_country_lengths.rs"]
mod country_lengths;

static START: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z]{2}[0-9]{2} ").expect("IBAN prefix regex"));
static WORD_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A\w").expect("Unicode word suffix regex"));

/// Byte spans in the original input, including single ASCII grouping spaces.
/// The registered length determines the endpoint before checksum validation.
/// A space after that endpoint starts separate text, including numeric text.
/// This does not validate account existence or country-specific BBAN rules.
pub(super) fn candidates(text: &str) -> impl Iterator<Item = Range<usize>> + '_ {
    let mut previous_end = 0;
    START.find_iter(text).filter_map(move |prefix| {
        let start = prefix.start();
        if start < previous_end {
            return None;
        }
        let mut remaining = country_lengths::length(&text[start..start + 2])?;
        let mut end = start;
        while remaining > 0 {
            let width = remaining.min(4);
            let group = text.as_bytes().get(end..end + width)?;
            if !group
                .iter()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
            {
                return None;
            }
            end += width;
            remaining -= width;
            if remaining > 0 {
                if text.as_bytes().get(end) != Some(&b' ') {
                    return None;
                }
                end += 1;
            }
        }
        // In particular, do not truncate a longer final group or detach a
        // combining mark. Unicode word boundaries match the compact scanner.
        if WORD_SUFFIX.is_match(&text[end..]) || !super::iban_mod97_valid(&text[start..end]) {
            return None;
        }
        previous_end = end;
        Some(start..end)
    })
}
