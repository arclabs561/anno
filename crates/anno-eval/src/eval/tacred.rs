//! Native, locally supplied TACRED splits and the author's patched Re-TACRED splits.

use super::dataset_registry::DatasetId;
use super::loader::{AnnotatedSentence, AnnotatedToken, RelationDocument};
use super::relation::RelationGold;
use anno::{Error, Result};
use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct Record {
    token: Vec<String>,
    relation: String,
    subj_start: usize,
    subj_end: usize,
    subj_type: String,
    obj_start: usize,
    obj_end: usize,
    obj_type: String,
}

pub(super) fn records(content: &str) -> Result<Vec<Record>> {
    let invalid_json = |error| Error::InvalidInput(format!("Invalid TACRED split: {error}"));
    let records: Vec<Record> = if content.trim_start().starts_with('[') {
        serde_json::from_str(content).map_err(invalid_json)?
    } else {
        // Also accept the one-record-per-line representation used by local exports.
        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<serde_json::Result<_>>()
            .map_err(invalid_json)?
    };
    if records.is_empty() {
        return Err(Error::InvalidInput(
            "TACRED split contains no records".into(),
        ));
    }
    for (i, record) in records.iter().enumerate() {
        if record.subj_start > record.subj_end
            || record.subj_end >= record.token.len()
            || record.obj_start > record.obj_end
            || record.obj_end >= record.token.len()
        {
            return Err(Error::InvalidInput(format!(
                "Invalid subject or object token range in TACRED record {}",
                i + 1
            )));
        }
        if record.relation.trim().is_empty()
            || record.subj_type.trim().is_empty()
            || record.obj_type.trim().is_empty()
            || record.token.iter().any(String::is_empty)
        {
            return Err(Error::InvalidInput(format!(
                "Empty token or label in TACRED record {}",
                i + 1
            )));
        }
    }
    Ok(records)
}

impl Record {
    pub(super) fn sentence(self, id: DatasetId) -> AnnotatedSentence {
        // Flat BIO can represent each designated argument when they are disjoint.
        // If they overlap, preserve the complete subject rather than splitting a mention.
        let disjoint = self.obj_end < self.subj_start || self.subj_end < self.obj_start;
        let tokens = self
            .token
            .into_iter()
            .enumerate()
            .map(|(i, text)| {
                let argument = if (self.subj_start..=self.subj_end).contains(&i) {
                    Some((self.subj_start, &self.subj_type))
                } else if disjoint && (self.obj_start..=self.obj_end).contains(&i) {
                    Some((self.obj_start, &self.obj_type))
                } else {
                    None
                };
                let ner_tag = match argument {
                    Some((start, label)) => {
                        format!("{}-{label}", if start == i { "B" } else { "I" })
                    }
                    None => "O".into(),
                };
                AnnotatedToken { text, ner_tag }
            })
            .collect();
        AnnotatedSentence {
            tokens,
            source_dataset: id,
        }
    }

    pub(super) fn document(self) -> RelationDocument {
        // The source endpoints include the last token. Anno uses half-open character spans
        // in the sentence reconstructed by joining the original tokens with one space.
        let argument = |first: usize, last: usize| {
            let start = self.token[..first]
                .iter()
                .map(|t| t.chars().count() + 1)
                .sum::<usize>();
            let text = self.token[first..=last].join(" ");
            ((start, start + text.chars().count()), text)
        };
        let relations = if self.relation == "no_relation" {
            vec![]
        } else {
            let (subject, head) = argument(self.subj_start, self.subj_end);
            let (object, tail) = argument(self.obj_start, self.obj_end);
            vec![RelationGold::new(
                subject,
                self.subj_type,
                head,
                object,
                self.obj_type,
                tail,
                self.relation,
            )]
        };
        RelationDocument {
            text: self.token.join(" "),
            relations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::loader::{DatasetLoader, LoadableDatasetId};
    use serde_json::json;

    // Authored records with the object before the subject and Unicode in both arguments.
    fn example() -> serde_json::Value {
        json!({"token":["Café","Éclat","hired","李","."],
            "subj_start":3,"subj_end":3,"subj_type":"PERSON",
            "obj_start":0,"obj_end":1,"obj_type":"ORGANIZATION","relation":"per:employee_of"})
    }

    #[test]
    fn local_tacred_splits_preserve_direction_offsets_and_negative_examples() {
        let cache = tempfile::tempdir().unwrap();
        let loader = DatasetLoader::with_cache_dir(cache.path()).unwrap();
        let positive = example();
        let mut negative = example();
        negative["relation"] = json!("no_relation");
        let array = serde_json::to_string_pretty(&[&positive, &negative]).unwrap();
        let lines = format!("{positive}\n{negative}\n");
        for (id, split) in [(DatasetId::ReTACRED, array), (DatasetId::TACRED, lines)] {
            assert!(id.requires_local_source() && id.excluded_from_shared_cache());
            #[cfg(feature = "eval")]
            assert!(loader
                .load_or_download_relation(id)
                .unwrap_err()
                .to_string()
                .contains("authorized local source"));
            std::fs::write(cache.path().join(id.cache_filename()), split).unwrap();
            let docs = loader.load_relation(id).unwrap();
            assert_eq!(docs.len(), 2);
            assert_eq!(docs[0].text, "Café Éclat hired 李 .");
            let [relation] = &docs[0].relations[..] else {
                panic!("expected one relation")
            };
            assert_eq!(
                (
                    relation.head_text.as_str(),
                    relation.head_span,
                    relation.head_type.as_str()
                ),
                ("李", (17, 18), "PERSON")
            );
            assert_eq!(
                (
                    relation.tail_text.as_str(),
                    relation.tail_span,
                    relation.tail_type.as_str()
                ),
                ("Café Éclat", (0, 10), "ORGANIZATION")
            );
            assert_eq!(relation.relation_type, "per:employee_of");
            assert!(docs[1].relations.is_empty());
            let sentences = loader
                .load(LoadableDatasetId::try_from(id).unwrap())
                .unwrap()
                .sentences;
            assert_eq!(sentences[1].text(), docs[1].text);
            let tags = sentences[1]
                .tokens
                .iter()
                .map(|t| t.ner_tag.as_str())
                .collect::<Vec<_>>();
            assert_eq!(
                tags,
                ["B-ORGANIZATION", "I-ORGANIZATION", "O", "B-PERSON", "O"]
            );
        }
    }

    #[test]
    fn tacred_rejects_bad_ranges_and_label_only_patches() {
        for (key, value) in [("subj_start", 2), ("obj_end", 80)] {
            let mut bad = example();
            if key == "subj_start" {
                bad["subj_end"] = json!(0);
            }
            bad[key] = json!(value);
            assert!(records(&json!([example(), bad]).to_string()).is_err());
        }
        for input in [r#"{"authored-id":"per:employee_of"}"#, "[]", "", "{}"] {
            assert!(records(input).is_err());
        }
    }
}
