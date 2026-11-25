//! Normalize the authors' processed JSONL split to sentence-relative CrossRE rows.

use anno::{Error, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{cmp::Reverse, collections::HashSet};

type Mention = (usize, usize, String);
type Relation = (usize, usize, usize, usize, String);

#[derive(Deserialize)]
struct Abstract {
    sentences: Vec<Vec<String>>,
    ner: Vec<Vec<Mention>>,
    relations: Vec<Vec<Relation>>,
}

pub(super) fn sentences(content: &str, flat_ner: bool) -> Result<Vec<Value>> {
    let mut rows = Vec::new();
    for (line, text) in content
        .lines()
        .enumerate()
        .filter(|(_, t)| !t.trim().is_empty())
    {
        let invalid =
            |message: &str| Error::InvalidInput(format!("SciERC line {}: {message}", line + 1));
        let doc: Abstract = serde_json::from_str(text).map_err(|e| invalid(&e.to_string()))?;
        if doc.sentences.is_empty()
            || doc.sentences.len() != doc.ner.len()
            || doc.sentences.len() != doc.relations.len()
        {
            return Err(invalid(
                "expected aligned sentence, NER and relation groups",
            ));
        }
        let mut offset = 0;
        for ((tokens, mut mentions), mut relations) in
            doc.sentences.into_iter().zip(doc.ner).zip(doc.relations)
        {
            if tokens.is_empty() || tokens.iter().any(String::is_empty) {
                return Err(invalid("empty sentence or token"));
            }
            let end = offset + tokens.len();
            let mut spans = HashSet::new();
            for (first, last, label) in &mut mentions {
                if label.is_empty()
                    || *first < offset
                    || *first > *last
                    || *last >= end
                    || !spans.insert((*first, *last))
                {
                    return Err(invalid("invalid or duplicate entity span"));
                }
                *first -= offset;
                *last -= offset;
            }
            for (h0, h1, t0, t1, label) in &mut relations {
                if label.is_empty() || !spans.contains(&(*h0, *h1)) || !spans.contains(&(*t0, *t1))
                {
                    return Err(invalid(
                        "relation argument has no entity in its sentence or label is empty",
                    ));
                }
                *h0 -= offset;
                *h1 -= offset;
                *t0 -= offset;
                *t1 -= offset;
            }
            if flat_ner {
                // One BIO sequence can represent only disjoint mentions. Keep complete
                // leftmost mentions and prefer the longer span when starts coincide.
                mentions.sort_unstable_by_key(|(first, last, _)| (*first, Reverse(*last)));
                let mut next = 0;
                mentions.retain(|(first, last, _)| {
                    if *first < next {
                        return false;
                    }
                    next = last + 1;
                    true
                });
            }
            rows.push(json!({"sentence":tokens,"ner":mentions,"relations":relations}));
            offset = end;
        }
    }
    if rows.is_empty() {
        return Err(Error::InvalidInput(
            "SciERC split contains no sentences".into(),
        ));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{dataset_registry::DatasetId, loader::DatasetLoader};

    fn authored_abstract() -> Value {
        json!({"sentences":[["Résumé","."],["Éma","lens","works","."]],
            "ner":[[],[[2,3,"Method"],[3,3,"Material"],[4,4,"Task"]]],
            "relations":[[],[[3,3,4,4,"USED-FOR"]]]})
    }

    #[test]
    fn scierc_rebases_later_sentences_and_retains_nested_relation_arguments() {
        let cache = tempfile::tempdir().unwrap();
        let loader = DatasetLoader::with_cache_dir(cache.path()).unwrap();
        let id = DatasetId::SciERC;
        let input = authored_abstract().to_string();
        std::fs::write(cache.path().join(id.cache_filename()), &input).unwrap();
        let docs = loader.load_or_download_relation(id).unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs[0].relations.is_empty());
        assert_eq!(docs[1].text, "Éma lens works .");
        let r = &docs[1].relations[0];
        assert_eq!(
            (r.head_text.as_str(), r.head_type.as_str(), r.head_span),
            ("lens", "Material", (4, 8))
        );
        assert_eq!(
            (r.tail_text.as_str(), r.tail_type.as_str(), r.tail_span),
            ("works", "Task", (9, 14))
        );
        assert_eq!(r.relation_type, "USED-FOR");
        let flat = loader.parse_content_str(&input, id).unwrap();
        let entities = flat.sentences[1].entities();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "Éma lens");
    }

    #[test]
    fn scierc_rejects_misaligned_groups_and_ungrounded_relations() {
        let mut bad = authored_abstract();
        bad["relations"][1][0][0] = json!(1);
        assert!(sentences(&bad.to_string(), false).is_err());
        bad["relations"] = json!([]);
        assert!(sentences(&bad.to_string(), true).is_err());
    }
}
