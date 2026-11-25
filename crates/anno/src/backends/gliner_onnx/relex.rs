//! Batch-one joint inference for the trained GLiNER-Relex directed pair head.
use super::{GLiNEROnnx, DEFAULT_GLINER_LABELS, MAX_SPAN_WIDTH};
use crate::backends::inference::{ExtractionWithRelations, RelationExtractor, RelationTriple};
use crate::{Confidence, Entity, Error, Result, SpanConverter};
use ort::value::Tensor;
use regex::Regex;
use std::sync::LazyLock;

static WORDS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\p{L}\p{N}_]+(?:[-_][\p{L}\p{N}_]+)*|\S").expect("valid upstream word splitter")
});

fn inference_error(e: impl std::fmt::Display) -> Error {
    Error::Inference(format!("GLiNER-Relex: {e}"))
}

impl RelationExtractor for GLiNEROnnx {
    fn extract_with_relations(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        if !self.supports_relex_relations() {
            return Err(Error::FeatureNotAvailable(
                "Trained Relex relations require scripts/export_gliner_relex_onnx.py --joint"
                    .into(),
            ));
        }
        let entities = if entity_types.is_empty() {
            DEFAULT_GLINER_LABELS
        } else {
            entity_types
        };
        self.extract_joint(text, entities, relation_types, threshold)
    }
}

impl GLiNEROnnx {
    pub(super) fn extract_joint(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::InvalidInput(
                "Relex threshold must be between 0 and 1".into(),
            ));
        }
        let words: Vec<_> = WORDS.find_iter(text).collect();
        if words.is_empty() || entity_types.is_empty() {
            return Ok(Default::default());
        }
        // Markers and labels are upstream pretokenized words. Word IDs (not
        // token-ID equality) distinguish markers from literal source text.
        let mut prompt = Vec::new();
        let mut entity_words = Vec::new();
        let mut relation_words = Vec::new();
        for &label in entity_types {
            entity_words.push(prompt.len());
            prompt.extend(["<<ENT>>", label]);
        }
        prompt.push("<<SEP>>");
        for &label in relation_types {
            relation_words.push(prompt.len());
            prompt.extend(["<<REL>>", label]);
        }
        prompt.push("<<SEP>>");
        let prefix = prompt.len();
        prompt.extend(words.iter().map(|w| w.as_str()));
        let n_prompt = prompt.len();
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(inference_error)?;
        let mut positions = vec![None; n_prompt];
        for (i, word) in enc.get_word_ids().iter().enumerate() {
            if let Some(word) = word {
                positions[*word as usize].get_or_insert(i as i64);
            }
        }
        let positions = positions
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                Error::InvalidInput(
                    "Relex tokenizer truncated or omitted a prompt or source word".into(),
                )
            })?;
        let input_ids: Vec<i64> = enc.get_ids().iter().map(|&v| i64::from(v)).collect();
        let count = words.len();
        let (span_idx, span_mask) = self.make_span_tensors(count);
        let slots = span_mask.len();
        let idx = |indices: &[usize]| indices.iter().map(|&i| positions[i]).collect::<Vec<i64>>();
        let vector = |v: Vec<i64>| Tensor::from_array(([v.len()], v)).map_err(inference_error);
        // ONNX MatMul implementations vary in support for zero-sized classes.
        // A dummy embedding is discarded when the requested relation schema is empty.
        let pair_threshold = if relation_types.is_empty() {
            1.0
        } else {
            threshold
        };
        let rel_pos = if relation_words.is_empty() {
            vec![0]
        } else {
            idx(&relation_words)
        };
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array(([1, input_ids.len()], input_ids)).map_err(inference_error)?,
            "word_positions" => vector(positions[prefix..].to_vec())?,
            "entity_positions" => vector(idx(&entity_words))?,
            "relation_positions" => vector(rel_pos)?,
            "span_idx" => Tensor::from_array(([1, slots, 2], span_idx)).map_err(inference_error)?,
            "span_mask" => Tensor::from_array(([slots], span_mask)).map_err(inference_error)?,
            "threshold" => Tensor::from_array(([] as [usize;0], vec![pair_threshold])).map_err(inference_error)?
        ]).map_err(inference_error)?;
        let (dims, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(inference_error)?;
        if dims.as_ref()
            != [
                1,
                count as i64,
                MAX_SPAN_WIDTH as i64,
                entity_types.len() as i64,
            ]
        {
            return Err(inference_error(format!(
                "unexpected entity logits shape {dims:?}"
            )));
        }
        let mut candidates = Vec::new();
        for (slot, classes) in logits.chunks_exact(entity_types.len()).enumerate() {
            let start = slot / MAX_SPAN_WIDTH;
            let end = start + slot % MAX_SPAN_WIDTH;
            if end >= count {
                continue;
            }
            for (class, &logit) in classes.iter().enumerate() {
                let score = 1.0 / (1.0 + (-logit).exp());
                if score > threshold {
                    candidates.push((slot, class, score));
                }
            }
        }
        // Same flat greedy overlap policy and tie order as upstream. Keep the
        // original grid slot so relation endpoints survive reordering and NMS.
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
        let mut occupied = vec![false; count];
        candidates.retain(|&(slot, _, _)| {
            let start = slot / MAX_SPAN_WIDTH;
            let end = start + slot % MAX_SPAN_WIDTH;
            if occupied[start..=end].contains(&true) {
                return false;
            }
            occupied[start..=end].fill(true);
            true
        });
        candidates.sort_by_key(|v| v.0);
        let offsets = SpanConverter::new(text);
        let mut slot_to_entity = vec![None; slots];
        let mut entities = Vec::new();
        for (slot, label, score) in candidates {
            let first = slot / MAX_SPAN_WIDTH;
            let start = words[first].start();
            let end = words[first + slot % MAX_SPAN_WIDTH].end();
            slot_to_entity[slot] = Some(entities.len());
            entities.push(Entity::new(
                &text[start..end],
                Self::map_entity_type(entity_types[label]),
                offsets.byte_to_char(start),
                offsets.byte_to_char(end),
                f64::from(score),
            ));
        }
        let mut relations = Vec::new();
        if !relation_types.is_empty() {
            let (pd, pairs) = outputs["pair_slots"]
                .try_extract_tensor::<i64>()
                .map_err(inference_error)?;
            let (sd, scores) = outputs["relation_scores"]
                .try_extract_tensor::<f32>()
                .map_err(inference_error)?;
            if pd.len() != 2 || pd[1] != 2 || sd.as_ref() != [pd[0], relation_types.len() as i64] {
                return Err(inference_error(
                    "invalid directed relation output dimensions",
                ));
            }
            for (pair, row) in pairs
                .chunks_exact(2)
                .zip(scores.chunks_exact(relation_types.len()))
            {
                let endpoint = |slot: i64| {
                    slot_to_entity
                        .get(slot as usize)
                        .copied()
                        .ok_or_else(|| inference_error("relation endpoint outside the span grid"))
                };
                let (Some(head_idx), Some(tail_idx)) = (endpoint(pair[0])?, endpoint(pair[1])?)
                else {
                    continue;
                };
                for (&score, &label) in row.iter().zip(relation_types) {
                    if score > threshold {
                        relations.push(RelationTriple {
                            head_idx,
                            tail_idx,
                            relation_type: label.to_owned(),
                            confidence: Confidence::new(f64::from(score)),
                        });
                    }
                }
            }
        }
        Ok(ExtractionWithRelations {
            entities,
            relations,
        })
    }
}
