//! Candidate boundaries for printed IBANs; compact recognition remains separate.

use regex::Regex;
use std::{ops::Range, sync::LazyLock};

#[path = "iban_country_lengths.rs"]
mod country_lengths;

static START: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z]{2}[0-9]{2}[ -]").expect("IBAN prefix regex"));
static WORD_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A\w").expect("Unicode word suffix regex"));

/// Consume the standard four-character grouping: uppercase alphanumeric
/// groups of four separated by single ASCII spaces, with a final group of
/// one to four characters.
fn four_char_groups(text: &str, start: usize, length: usize) -> Option<usize> {
    let mut remaining = length;
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
    Some(end)
}

/// Consume registry print groupings. Some countries group by account
/// component rather than in fours (BI 4,5,5,11,2; LY 4,3,3,15; SV 2,2,4,20),
/// so group widths are not capped. The bounded invariants are one or more
/// uppercase/digit groups, the same single ASCII space or hyphen between
/// groups, and exactly the registered non-separator length.
fn registry_groups(text: &str, start: usize, length: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut end = start;
    let mut consumed = 0;
    let mut separator = None;
    while consumed < length {
        let first = *bytes.get(end)?;
        if !(first.is_ascii_uppercase() || first.is_ascii_digit()) {
            return None;
        }
        while consumed < length {
            let byte = *bytes.get(end)?;
            if byte.is_ascii_uppercase() || byte.is_ascii_digit() {
                end += 1;
                consumed += 1;
            } else if byte == b' ' || byte == b'-' {
                break;
            } else {
                return None;
            }
        }
        if consumed < length {
            let byte = *bytes.get(end)?;
            match separator {
                Some(previous) if previous != byte => return None,
                None => separator = Some(byte),
                _ => {}
            }
            end += 1;
        }
    }
    Some(end)
}

/// Byte spans in the original input, including single ASCII grouping spaces
/// or hyphens. The registered length determines the endpoint before checksum
/// validation. A separator after that endpoint starts separate text, including
/// numeric text. This does not validate account existence or country-specific
/// BBAN rules.
pub(super) fn candidates(text: &str) -> impl Iterator<Item = Range<usize>> + '_ {
    let mut previous_end = 0;
    START.find_iter(text).filter_map(move |prefix| {
        let start = prefix.start();
        if start < previous_end {
            return None;
        }
        let length = country_lengths::length(&text[start..start + 2])?;
        let end = four_char_groups(text, start, length)
            .or_else(|| registry_groups(text, start, length))?;
        // In particular, do not truncate a longer final group or detach a
        // combining mark. Unicode word boundaries match the compact scanner.
        if WORD_SUFFIX.is_match(&text[end..]) || !super::iban_mod97_valid(&text[start..end]) {
            return None;
        }
        previous_end = end;
        Some(start..end)
    })
}
