//! Named entities and surface spacing in the pinned NorNE UD release.

use super::{AnnotatedSentence, AnnotatedToken, DatasetId, Error, Result, NORNE_UD_URL};

/// Resolve only the six publisher files whose language and split are known.
pub(super) fn profile(url: &str) -> Option<(&'static str, &'static str)> {
    let name = url.strip_prefix(NORNE_UD_URL)?.strip_suffix(".conllu")?;
    let (language, split) = if let Some(split) = name.strip_prefix("nno/no_nynorsk-ud-") {
        ("nn", split)
    } else {
        ("nb", name.strip_prefix("nob/no_bokmaal-ud-")?)
    };
    ["train", "dev", "test"]
        .into_iter()
        .find(|&s| s == split)
        .map(|s| (language, s))
}

fn is_bio_tag(tag: &str) -> bool {
    tag == "O"
        || tag
            .strip_prefix("B-")
            .or_else(|| tag.strip_prefix("I-"))
            .is_some_and(|label| !label.is_empty() && !label.chars().any(char::is_whitespace))
}

pub(super) fn sentences(content: &str) -> Result<Vec<AnnotatedSentence>> {
    let mut sentences = Vec::new();
    let mut tokens: Vec<AnnotatedToken> = Vec::new();
    let mut space_after = Vec::new();
    let mut source_text = None;
    for (index, line) in content.lines().chain(std::iter::once("")).enumerate() {
        let invalid = |reason: &str| {
            Error::InvalidInput(format!(
                "Invalid NorNE CoNLL-U at line {}: {reason}",
                index + 1
            ))
        };
        if line.is_empty() {
            if tokens.is_empty() && source_text.is_none() {
                continue;
            }
            let source = source_text
                .take()
                .ok_or_else(|| invalid("missing # text header"))?;
            let sentence = AnnotatedSentence::from_tokens_with_spacing(
                DatasetId::NorNE,
                std::mem::take(&mut tokens),
                std::mem::take(&mut space_after),
            )?;
            if sentence.tokens.is_empty() || sentence.text() != source {
                return Err(invalid("word forms and SpaceAfter disagree with # text"));
            }
            sentences.push(sentence);
        } else if let Some(text) = line.strip_prefix("# text = ") {
            if source_text.replace(text).is_some() {
                return Err(invalid("duplicate # text header"));
            }
        } else if !line.starts_with('#') {
            let fields: [&str; 10] = line
                .split('\t')
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| invalid("expected ten tab-separated columns"))?;
            // The pinned release annotates consecutive integer-indexed words.
            // Ranges or empty nodes need separate alignment with the surface form.
            if fields[0].parse::<usize>().ok() != Some(tokens.len() + 1) || fields[1].is_empty() {
                return Err(invalid("expected the next numbered word and its form"));
            }
            let mut labels = fields[9]
                .split('|')
                .filter_map(|attr| attr.strip_prefix("name="));
            let label = labels
                .next()
                .filter(|label| is_bio_tag(label))
                .ok_or_else(|| invalid("expected a name= BIO tag in MISC"))?;
            if labels.next().is_some() {
                return Err(invalid("duplicate name= tag in MISC"));
            }
            if let Some(ty) = label.strip_prefix("I-") {
                let previous = tokens.last().and_then(|word| word.ner_tag.split_once('-'));
                if !matches!(previous, Some(("B" | "I", previous_ty)) if previous_ty == ty) {
                    return Err(invalid("name=I tag has no matching preceding entity"));
                }
            }
            space_after.push(!fields[9].split('|').any(|attr| attr == "SpaceAfter=No"));
            tokens.push(AnnotatedToken {
                text: fields[1].to_owned(),
                ner_tag: label.to_owned(),
            });
        }
    }
    Ok(sentences)
}

#[cfg(test)]
mod tests {
    use super::super::{DatasetLoader, LoadableDatasetId};
    use super::*;

    // A constructed punctuation and supplementary-plane fixture.
    const ORIGINAL: &str = "\
# sent_id = fixture-1
# text = Å,😀 X-Y!
1\tÅ\t_\t_\t_\t_\t_\t_\t_\tname=B-PER|SpaceAfter=No
2\t,\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No|name=O
3\t😀\t_\t_\t_\t_\t_\t_\t_\tname=O
4\tX\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No|name=B-ORG
5\t-\t_\t_\t_\t_\t_\t_\t_\tname=I-ORG|SpaceAfter=No
6\tY\t_\t_\t_\t_\t_\t_\t_\tname=I-ORG|SpaceAfter=No
7\t!\t_\t_\t_\t_\t_\t_\t_\tname=O";

    #[test]
    fn source_spacing_survives_cache_loading_and_serialization() {
        let root = tempfile::tempdir().unwrap();
        let loader = DatasetLoader::with_cache_dir(root.path()).unwrap();
        let id = LoadableDatasetId::try_from(DatasetId::NorNE).unwrap();
        std::fs::write(loader.cache_path(id), ORIGINAL.replace('\n', "\r\n")).unwrap();
        let data = loader.load(id).unwrap();
        assert!(data.text_coordinate_note().unwrap().contains("Publisher"));
        assert_eq!(data.stats().tokens, 7);
        let s = &data.sentences[0];
        let encoded = serde_json::to_value(s).unwrap();
        let reopened: AnnotatedSentence = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(
            reopened.source_space_after(),
            Some([false, false, true, false, false, false, true].as_slice())
        );
        assert_eq!(reopened.text(), "Å,😀 X-Y!");
        let gold = reopened.entities();
        let covered: Vec<_> = gold
            .iter()
            .map(|g| (g.text.as_str(), g.original_label.as_str(), g.start, g.end))
            .collect();
        assert_eq!(covered, [("Å", "PER", 0, 1), ("X-Y", "ORG", 4, 7)]);
        let mut malformed = encoded;
        malformed["space_after"] = serde_json::json!([true, false]);
        assert!(serde_json::from_value::<AnnotatedSentence>(malformed).is_err());
    }

    #[test]
    fn legacy_bio_and_old_serialized_sentences_keep_normalized_coordinates() {
        let legacy = "Å\tB-PER\n,\tO\n😀\tO\nX\tB-ORG\n-\tI-ORG\nY\tI-ORG\n!\tO\n";
        let loader = DatasetLoader::new().unwrap();
        let data = loader.parse_content_str(legacy, DatasetId::NorNE).unwrap();
        assert!(data.text_coordinate_note().unwrap().contains("legacy"));
        let sentence = &data.sentences[0];
        let encoded = serde_json::to_value(sentence).unwrap();
        assert!(encoded.get("space_after").is_none());
        let reopened: AnnotatedSentence = serde_json::from_value(encoded).unwrap();
        assert!(reopened.source_space_after().is_none());
        assert_eq!(reopened.text(), "Å , 😀 X - Y !");
        let org = &reopened.entities()[1];
        assert_eq!((org.text.as_str(), org.start, org.end), ("X - Y", 6, 11));
    }

    #[test]
    fn older_word_pos_chunk_tag_inputs_still_use_their_word_and_tag_columns() {
        let loader = DatasetLoader::new().unwrap();
        let input = "-DOCSTART- -X- O O\n\nÅ N B-NP B-PER\n! P O O";
        let data = loader.parse_content_str(input, DatasetId::NorNE).unwrap();
        assert_eq!(data.sentences.len(), 1);
        assert_eq!(data.sentences[0].text(), "Å !");
        assert_eq!(data.sentences[0].entities()[0].original_label, "PER");
        assert!(data.text_coordinate_note().unwrap().contains("legacy"));
    }

    #[test]
    fn invalid_word_rows_or_spacing_never_become_scored_gold() {
        let loader = DatasetLoader::new().unwrap();
        for input in [
            ORIGINAL.replace("# text = Å,", "# text = Å ,"),
            ORIGINAL.replace("name=B-PER", "name=B-PER|name=O"),
            ORIGINAL.replace("name=B-PER", "name=B-"),
            ORIGINAL.replace("name=B-PER", "name=I-PER"),
            ORIGINAL.replace("name=B-PER", "_"),
            ORIGINAL.replace("2\t,", "2-3\t,"),
            ORIGINAL.replace("# text = Å,😀 X-Y!\n", ""),
            format!("a\tO\n{ORIGINAL}"),
        ] {
            assert!(loader.parse_content_str(&input, DatasetId::NorNE).is_err());
        }
    }

    #[test]
    fn provenance_is_limited_to_the_six_pinned_profiles() {
        for (stem, lang) in [("nno/no_nynorsk", "nn"), ("nob/no_bokmaal", "nb")] {
            for split in ["train", "dev", "test"] {
                let url = format!("{NORNE_UD_URL}{stem}-ud-{split}.conllu");
                assert_eq!(profile(&url), Some((lang, split)));
            }
        }
        assert_eq!(
            profile(&format!("{NORNE_UD_URL}nno/no_bokmaal-ud-test.conllu")),
            None
        );
        assert_eq!(
            profile("https://example.org/nno/no_nynorsk-ud-test.conllu"),
            None
        );
    }
}
