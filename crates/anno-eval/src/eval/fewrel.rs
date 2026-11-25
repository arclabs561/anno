//! FewRel's released Wikipedia validation file groups examples by Wikidata predicate.
//! The publisher's encoder takes the first listed mention of each designated entity.

use super::loader::{AnnotatedSentence, AnnotatedToken, DatasetId, RelationDocument};
use super::relation::RelationGold;
use anno::{Error, Result};
use serde::Deserialize;
use std::{collections::BTreeMap, ops::Range};

#[derive(Deserialize)]
struct Row {
    tokens: Vec<String>,
    h: Endpoint,
    t: Endpoint,
}

// The display name and Wikidata entity ID need not match the mention text.
type Endpoint = (
    serde::de::IgnoredAny,
    serde::de::IgnoredAny,
    Vec<Vec<usize>>,
);

pub(super) struct Example {
    tokens: Vec<String>,
    predicate: String,
    head: Range<usize>,
    tail: Range<usize>,
}

fn mention(endpoint: &Endpoint, tokens: usize) -> Result<Range<usize>> {
    let invalid = || {
        Error::InvalidInput(
            "FewRel requires contiguous in-bounds token indices for each designated mention".into(),
        )
    };
    let indices = endpoint.2.first().ok_or_else(invalid)?;
    let (&start, &end) = indices.first().zip(indices.last()).ok_or_else(invalid)?;
    if end >= tokens
        || !indices
            .windows(2)
            .all(|pair| pair[0].checked_add(1) == Some(pair[1]))
    {
        return Err(invalid());
    }
    Ok(start..end + 1)
}

pub(super) fn read(content: &str) -> Result<Vec<Example>> {
    let groups: BTreeMap<String, Vec<Row>> = serde_json::from_str(content)
        .map_err(|e| Error::InvalidInput(format!("Invalid FewRel predicate-grouped JSON: {e}")))?;
    let mut examples = Vec::new();
    for (predicate, rows) in groups {
        if predicate.is_empty() {
            return Err(Error::InvalidInput("Empty FewRel predicate".into()));
        }
        for row in rows {
            let head = mention(&row.h, row.tokens.len())?;
            let tail = mention(&row.t, row.tokens.len())?;
            if head.start < tail.end && tail.start < head.end {
                return Err(Error::InvalidInput(
                    "Overlapping FewRel designated mentions".into(),
                ));
            }
            examples.push(Example {
                tokens: row.tokens,
                predicate: predicate.clone(),
                head,
                tail,
            });
        }
    }
    if examples.is_empty() {
        return Err(Error::InvalidInput(
            "FewRel contains no relation examples".into(),
        ));
    }
    Ok(examples)
}

impl Example {
    pub(super) fn into_sentence(self) -> AnnotatedSentence {
        let mut tokens: Vec<_> = self
            .tokens
            .into_iter()
            .map(|text| AnnotatedToken {
                text,
                ner_tag: "O".into(),
            })
            .collect();
        for (span, role) in [(self.head, "head"), (self.tail, "tail")] {
            for (i, token) in tokens[span].iter_mut().enumerate() {
                token.ner_tag = format!("{}-{role}", if i == 0 { "B" } else { "I" });
            }
        }
        AnnotatedSentence {
            tokens,
            source_dataset: DatasetId::FewRel,
        }
    }

    pub(super) fn into_document(self) -> RelationDocument {
        let endpoint = |span: Range<usize>| {
            let start = self.tokens[..span.start]
                .iter()
                .map(|t| t.chars().count() + 1)
                .sum::<usize>();
            let surface = self.tokens[span].join(" ");
            ((start, start + surface.chars().count()), surface)
        };
        let (head, head_text) = endpoint(self.head);
        let (tail, tail_text) = endpoint(self.tail);
        RelationDocument {
            text: self.tokens.join(" "),
            relations: vec![RelationGold::new(
                head,
                "head",
                head_text,
                tail,
                "tail",
                tail_text,
                self.predicate,
            )],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::loader::{DatasetLoader, LoadableDatasetId};

    const EXAMPLE: &str = r#"{
        "P31": [{"tokens":["Élan","et","星","Bleu","revisit","Élan"],
            "h":["another display name","Q7",[[2,3]]], "t":["alias","Q8",[[0],[5]]]}],
        "P99": [{"tokens":["A","to","B"],"h":["A","Q1",[[0]]],"t":["B","Q2",[[2]]]}]
    }"#;

    #[test]
    fn fewrel_cache_loads_both_task_views() {
        let cache = tempfile::tempdir().unwrap();
        let id = DatasetId::FewRel;
        std::fs::write(cache.path().join(id.cache_filename()), EXAMPLE).unwrap();
        let loader = DatasetLoader::with_cache_dir(cache.path()).unwrap();
        let documents = loader.load_relation(id).unwrap();
        assert_eq!(documents.len(), 2);
        let doc = &documents[0];
        assert_eq!(doc.text, "Élan et 星 Bleu revisit Élan");
        assert_eq!(doc.relations.len(), 1);
        let rel = &doc.relations[0];
        assert_eq!(rel.relation_type, "P31");
        assert_eq!(rel.head_text, "星 Bleu");
        assert_eq!(rel.head_span, (8, 14));
        assert_eq!(rel.tail_text, "Élan");
        assert_eq!(rel.tail_span, (0, 4));
        assert_eq!(documents[1].relations[0].relation_type, "P99");
        let loaded = loader
            .load(LoadableDatasetId::try_from(id).unwrap())
            .unwrap();
        assert_eq!(loaded.len(), 2);
        let tags: Vec<_> = loaded.sentences[0]
            .tokens
            .iter()
            .map(|t| t.ner_tag.as_str())
            .collect();
        assert_eq!(tags, ["B-tail", "O", "B-head", "I-head", "O", "O"]);
    }

    #[test]
    fn fewrel_rejects_unrepresentable_designated_mentions() {
        for invalid in ["[[2,4]]", "[[4,3]]", "[[2,90]]", "[[0]]", "[[]]", "[]"] {
            assert!(
                read(&EXAMPLE.replace("[[2,3]]", invalid)).is_err(),
                "{invalid}"
            );
        }
        assert!(read("{}").is_err());
    }
}
