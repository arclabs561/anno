use super::DialogueTurn;
use crate::backends::coref::CorefCluster;
use crate::offset::SpanConverter;
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::LazyLock;

static FIRST_PERSON: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(i|me|my|mine|myself)\b").unwrap());

/// Resolve English first-person singular mentions using source-aligned speaker IDs.
///
/// Turns must be sorted, disjoint, and contain the exact source substring at their
/// half-open Unicode scalar offsets. Omit turns whose speaker is unknown. Gaps are
/// allowed. Unspecified language is treated as English. A turn containing quoted
/// speech is skipped in full. Speaker IDs are local to this document. Only groups
/// with at least two mentions are returned. No addressee is inferred.
pub fn resolve_speaker_mentions(
    text: &str,
    turns: &[DialogueTurn],
) -> crate::Result<Vec<CorefCluster>> {
    let map = SpanConverter::new(text);
    let length = text.chars().count();
    let mut last_end = 0;
    let mut speakers = BTreeMap::new();
    for (index, turn) in turns.iter().enumerate() {
        if turn.start < last_end || turn.start >= turn.end || turn.end > length {
            return Err(crate::Error::invalid_input(format!(
                "turn {index} has invalid or overlapping character offsets"
            )));
        }
        let start_byte = map.char_to_byte(turn.start);
        let slice = &text[start_byte..map.char_to_byte(turn.end)];
        if turn.text != slice || turn.speaker.trim().is_empty() {
            return Err(crate::Error::invalid_input(format!(
                "turn {index} needs a speaker ID and exact source text"
            )));
        }
        last_end = turn.end;
        let language = turn.language.as_deref().unwrap_or("en");
        let primary = language
            .split_once('-')
            .map_or(language, |(primary, _)| primary);
        if !primary.eq_ignore_ascii_case("en") || contains_quotation(slice) {
            continue;
        }
        for mention in FIRST_PERSON.find_iter(slice) {
            let mentions: &mut Vec<_> = speakers.entry(&turn.speaker).or_default();
            let start = map.byte_to_char(start_byte + mention.start());
            let end = map.byte_to_char(start_byte + mention.end());
            mentions.push((mention.as_str().to_owned(), (start, end)));
        }
    }
    Ok(speakers
        .into_iter()
        .filter(|(_, mentions)| mentions.len() >= 2)
        .enumerate()
        .map(|(id, (speaker, members))| {
            let (mentions, spans) = members.into_iter().unzip();
            CorefCluster {
                id: id as u32,
                canonical: speaker.clone(),
                mentions,
                spans,
            }
        })
        .collect())
}

// Internal apostrophes in contractions and possessives are not quote marks.
fn contains_quotation(text: &str) -> bool {
    let mut chars = text.chars().peekable();
    let mut previous = None::<char>;
    while let Some(ch) = chars.next() {
        match ch {
            '"' | '“' | '”' | '„' | '«' | '»' | '‘' | '`' => return true,
            '\'' | '’'
                if !previous.is_some_and(char::is_alphanumeric)
                    || !chars.peek().is_some_and(|next| next.is_alphanumeric()) =>
            {
                return true
            }
            _ => {}
        }
        previous = Some(ch);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_or_misaligned_turns_fail_without_results() {
        let valid = DialogueTurn::new("I spoke", "a").with_span(2, 9);
        let text = "é I spoke";
        for invalid in [
            DialogueTurn::new("I spoke", "a").with_span(3, 10),
            DialogueTurn::new("I agree", "a").with_span(2, 9),
            DialogueTurn::new("I spoke", " ").with_span(2, 9),
            DialogueTurn::new("", "a").with_span(2, 2),
        ] {
            assert!(resolve_speaker_mentions(text, &[invalid]).is_err());
        }
        assert!(resolve_speaker_mentions(text, &[valid.clone(), valid]).is_err());
    }
}
