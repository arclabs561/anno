//! Decode thresholded TPLinkerPlus tags from one document or window.
//! The upstream tagger orders the union of entity and predicate tag strings.
//! The numeric IDs of the separate entity and predicate vocabularies do not
//! determine the classifier column order.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A sparse handshaking matrix in the author's `get_spots` representation.
/// Token spans address Unicode scalar offsets in the supplied text.
#[derive(Debug, Deserialize)]
pub struct PlusPrediction {
    /// Text for this prediction.
    pub text: String,
    /// Half-open character offsets per actual token, with padding excluded.
    pub tok2char_span: Vec<[usize; 2]>,
    /// Predicate vocabulary from training.
    pub rel2id: BTreeMap<String, usize>,
    /// Entity vocabulary from training.
    pub entity_type2id: BTreeMap<String, usize>,
    /// Nonzero matrix entries (token start, token end inclusive, tag ID).
    pub spots: Vec<[usize; 3]>,
}

/// Entity output in the original decoder's document-local coordinates.
#[derive(Debug, PartialEq, Serialize)]
pub struct PlusEntity {
    /// Original entity type.
    #[serde(rename = "type")]
    pub label: String,
    /// Exact surface.
    pub text: String,
    /// Half-open token interval.
    pub tok_span: [usize; 2],
    /// Half-open character interval.
    pub char_span: [usize; 2],
}

/// A directed edge from matching head and tail tags. The upstream decoder
/// identifies endpoints by offsets and emits no confidence.
#[derive(Debug, PartialEq, Serialize)]
pub struct PlusRelation {
    /// Subject surface.
    pub subject: String,
    /// Object surface.
    pub object: String,
    /// Original predicate.
    pub predicate: String,
    /// Subject token span.
    pub subj_tok_span: [usize; 2],
    /// Object token span.
    pub obj_tok_span: [usize; 2],
    /// Subject character span.
    pub subj_char_span: [usize; 2],
    /// Object character span.
    pub obj_char_span: [usize; 2],
}

/// Upstream relation and entity lists in handshaking order.
#[derive(Debug, PartialEq, Serialize)]
pub struct PlusOutput {
    /// Directed relations.
    pub relation_list: Vec<PlusRelation>,
    /// Typed entity spans.
    pub entity_list: Vec<PlusEntity>,
}

const SEPARATOR: char = '\u{2e80}';

impl PlusPrediction {
    /// Classifier columns in upstream sorted Unicode order.
    pub fn tags(&self) -> Result<Vec<String>, String> {
        let mut tags = BTreeSet::new();
        for (vocab, kinds) in [
            (&self.rel2id, &["SH2OH", "OH2SH", "ST2OT", "OT2ST"][..]),
            (&self.entity_type2id, &["EH2ET"][..]),
        ] {
            for label in vocab.keys() {
                if label.contains(SEPARATOR) {
                    return Err(format!("label contains upstream tag separator: {label}"));
                }
                for kind in kinds {
                    tags.insert(format!("{label}{SEPARATOR}{kind}"));
                }
            }
        }
        Ok(tags.into_iter().collect())
    }

    /// Decode the active classifier columns, retaining shared or identical
    /// endpoint spans, reverse predicates, and overlapping mentions.
    pub fn decode(&self) -> Result<PlusOutput, String> {
        let mut offsets: Vec<_> = self.text.char_indices().map(|(byte, _)| byte).collect();
        offsets.push(self.text.len());
        let mut previous = [0, 0];
        for &[start, end] in &self.tok2char_span {
            if start >= end || end >= offsets.len() || start < previous[0] || end < previous[1] {
                return Err(format!("invalid token character interval {start}..{end}"));
            }
            previous = [start, end];
        }
        let tags = self.tags()?;
        // torch.nonzero on the upper triangle emits row-major distinct spots.
        let spots: BTreeSet<_> = self.spots.iter().copied().collect();
        let mut entities = Vec::new();
        let mut by_head: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut heads = Vec::new();
        let mut tails = BTreeSet::new();
        for [first, last, id] in spots {
            if first > last || last >= self.tok2char_span.len() {
                return Err(format!(
                    "invalid upper-triangle coordinates ({first}, {last})"
                ));
            }
            let tag = tags.get(id).ok_or_else(|| format!("unknown tag ID {id}"))?;
            let (label, kind) = tag.split_once(SEPARATOR).expect("tag constructed above");
            match kind {
                "EH2ET" => {
                    let lo = self.tok2char_span[first][0];
                    let hi = self.tok2char_span[last][1];
                    by_head.entry(first).or_default().push(entities.len());
                    entities.push(PlusEntity {
                        label: label.to_owned(),
                        text: self.text[offsets[lo]..offsets[hi]].to_owned(),
                        tok_span: [first, last + 1],
                        char_span: [lo, hi],
                    });
                }
                "SH2OH" => heads.push((label, first, last)),
                "OH2SH" => heads.push((label, last, first)),
                "ST2OT" => {
                    tails.insert((label, first, last));
                }
                "OT2ST" => {
                    tails.insert((label, last, first));
                }
                _ => unreachable!("known tag suffix"),
            }
        }
        let mut relations = Vec::new();
        for (predicate, subject, object) in heads {
            let (Some(subjects), Some(objects)) = (by_head.get(&subject), by_head.get(&object))
            else {
                continue;
            };
            for &s in subjects {
                let s = &entities[s];
                for &o in objects {
                    let o = &entities[o];
                    if tails.contains(&(predicate, s.tok_span[1] - 1, o.tok_span[1] - 1)) {
                        relations.push(PlusRelation {
                            subject: s.text.clone(),
                            object: o.text.clone(),
                            predicate: predicate.to_owned(),
                            subj_tok_span: s.tok_span,
                            obj_tok_span: o.tok_span,
                            subj_char_span: s.char_span,
                            obj_char_span: o.char_span,
                        });
                    }
                }
            }
        }
        Ok(PlusOutput {
            relation_list: relations,
            entity_list: entities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_upstream_handshaking_tagger() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/tplinker_plus_tags.json"))
                .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut input: PlusPrediction = serde_json::from_value(case["input"].clone()).unwrap();
            assert_eq!(serde_json::json!(input.tags().unwrap()), fixture["tags"]);
            assert_eq!(
                serde_json::json!(input.decode().unwrap()),
                case["output"],
                "{}",
                case["name"]
            );
            input.spots.push([1, 0, 0]);
            assert!(input.decode().is_err());
        }
    }
}
