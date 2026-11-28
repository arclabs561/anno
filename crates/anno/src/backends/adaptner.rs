//! Arabic nested named entities from the released AdaptNER AraBERTv02 checkpoint.
//!
//! Export once with `scripts/export_adaptner.py`. Each Wojood type has independent
//! begin and inside logits, so different types may occupy overlapping spans.

use super::hf_loader::{create_onnx_session, load_tokenizer};
use crate::{Entity, EntityType, Error, Language, Model, Result, SpanConverter};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use std::{ops::Range, path::Path, sync::Mutex};
use tokenizers::Tokenizer;

const REVISION: &str = "c3096335b1e8d00f51451e267b8034a0cc59876f";
const LABELS: [&str; 21] = [
    "CARDINAL", "CURR", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP",
    "OCC", "ORDINAL", "ORG", "PERCENT", "PERS", "PRODUCT", "QUANTITY", "TIME", "UNIT", "WEBSITE",
];
const CHANNELS: usize = LABELS.len() * 2;
const WINDOW: usize = 256;

#[derive(Deserialize)]
struct Manifest {
    revision: String,
    labels: Vec<String>,
}

/// Local FP32 multilabel BIO token classifier for Arabic text.
pub struct AdaptNer {
    tokenizer: Tokenizer,
    model: Mutex<Session>,
    cls: i64,
    sep: i64,
}

fn inference(error: impl std::fmt::Display) -> Error {
    Error::Inference(format!("AdaptNER: {error}"))
}

fn entity_type(label: &str) -> EntityType {
    let canonical = match label {
        "PERS" => EntityType::Person,
        "CURR" => EntityType::Money,
        "UNIT" => EntityType::Quantity,
        "WEBSITE" => EntityType::Url,
        _ => EntityType::from_label(label),
    };
    if canonical.as_label() == label {
        canonical
    } else {
        EntityType::custom(label, canonical.category())
    }
}

impl AdaptNer {
    /// Open the local export selected by `ADAPTNER_MODEL_DIR`.
    pub fn from_env() -> Result<Self> {
        let dir = std::env::var_os("ADAPTNER_MODEL_DIR").ok_or_else(|| {
            Error::ModelInit(
                "set ADAPTNER_MODEL_DIR to an export from scripts/export_adaptner.py".into(),
            )
        })?;
        Self::from_path(Path::new(&dir))
    }

    fn validate_manifest(dir: &Path) -> Result<()> {
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(dir.join("adaptner.json"))?)
            .map_err(|e| Error::ModelInit(format!("invalid AdaptNER manifest: {e}")))?;
        let labels = ["B", "I"]
            .into_iter()
            .flat_map(|prefix| LABELS.map(|t| format!("{prefix}-{t}")));
        if manifest.revision != REVISION || !manifest.labels.into_iter().eq(labels) {
            return Err(Error::ModelInit(
                "unsupported AdaptNER revision or BIO channel order".into(),
            ));
        }
        Ok(())
    }

    /// Check the local files and label manifest without initializing ONNX Runtime.
    pub fn model_files_available(dir: &Path) -> bool {
        dir.join("model.onnx").is_file()
            && dir.join("tokenizer.json").is_file()
            && Self::validate_manifest(dir).is_ok()
    }

    /// Validate the channel order and load the ONNX graph and its WordPiece tokenizer.
    pub fn from_path(dir: &Path) -> Result<Self> {
        Self::validate_manifest(dir)?;
        let mut tokenizer = load_tokenizer(&dir.join("tokenizer.json"))?;
        tokenizer.with_truncation(None).map_err(inference)?;
        tokenizer.with_padding(None);
        let special = |s| {
            tokenizer
                .token_to_id(s)
                .map(i64::from)
                .ok_or_else(|| Error::ModelInit(format!("AdaptNER tokenizer missing {s}")))
        };
        let (cls, sep) = (special("[CLS]")?, special("[SEP]")?);
        Ok(Self {
            tokenizer,
            cls,
            sep,
            model: Mutex::new(create_onnx_session(
                &dir.join("model.onnx"),
                Default::default(),
            )?),
        })
    }

    fn predict(&self, words: &[&str]) -> Result<Vec<[f32; CHANNELS]>> {
        // Like the checkpoint reference, tokenize with is_split_into_words=True and
        // take the first subword of each source word. Special tokens are added per window.
        let encoded = self.tokenizer.encode(words, false).map_err(inference)?;
        let ids = encoded.get_ids();
        let word_ids = encoded.get_word_ids();
        let mut first_logits = vec![[0.0; CHANNELS]; words.len()];
        let mut model = self.model.lock().map_err(inference)?;
        let mut start = 0;
        while start < ids.len() {
            let mut end = (start + WINDOW - 2).min(ids.len());
            while end < ids.len() && end > start && word_ids[end] == word_ids[end - 1] {
                end -= 1;
            }
            if start == end {
                return Err(Error::InvalidInput(
                    "AdaptNER word exceeds the 254-subword window".into(),
                ));
            }
            let input: Vec<_> = std::iter::once(self.cls)
                .chain(ids[start..end].iter().copied().map(i64::from))
                .chain(std::iter::once(self.sep))
                .collect();
            let seq = input.len();
            let input = Tensor::from_array(([1, seq], input)).map_err(inference)?;
            let mask = Tensor::from_array(([1, seq], vec![1i64; seq])).map_err(inference)?;
            let outputs = model
                .run(ort::inputs!["input_ids" => input, "attention_mask" => mask])
                .map_err(inference)?;
            let (shape, logits) = outputs
                .get("logits")
                .ok_or_else(|| inference("missing logits"))?
                .try_extract_tensor::<f32>()
                .map_err(inference)?;
            if shape.as_ref() != [1, seq as i64, CHANNELS as i64] {
                return Err(inference(format!("unexpected logit shape {shape:?}")));
            }
            for k in start..end {
                if k > start && word_ids[k] == word_ids[k - 1] {
                    continue;
                }
                let word = word_ids[k].ok_or_else(|| inference("unmapped subword"))? as usize;
                let row = (1 + k - start) * CHANNELS;
                let scores = &logits[row..row + CHANNELS];
                if !scores.iter().all(|s| s.is_finite()) {
                    return Err(inference("nonfinite token logits"));
                }
                first_logits
                    .get_mut(word)
                    .ok_or_else(|| inference("invalid word index"))?
                    .copy_from_slice(scores);
            }
            start = end;
        }
        Ok(first_logits)
    }
}

fn decode(text: &str, ranges: &[Range<usize>], logits: &[[f32; CHANNELS]]) -> Vec<Entity> {
    let offsets = SpanConverter::new(text);
    let mut entities = Vec::new();
    for (channel, label) in LABELS.iter().enumerate() {
        let mut active: Option<(usize, f32)> = None;
        for pos in 0..=logits.len() {
            let begins = pos < logits.len() && logits[pos][channel] > 0.0;
            let inside = pos < logits.len() && logits[pos][channel + LABELS.len()] > 0.0;
            if begins || !inside {
                if let Some((first, confidence)) = active.take() {
                    let bytes = ranges[first].start..ranges[pos - 1].end;
                    entities.push(Entity::new(
                        &text[bytes.clone()],
                        entity_type(label),
                        offsets.byte_to_char(bytes.start),
                        offsets.byte_to_char(bytes.end),
                        confidence,
                    ));
                }
            }
            if begins || inside {
                // B wins over I within a channel. Orphan I opens a new span.
                let score = logits[pos][if begins {
                    channel
                } else {
                    channel + LABELS.len()
                }];
                let probability = 1.0 / (1.0 + (-score).exp());
                let span = active.get_or_insert((pos, probability));
                span.1 = span.1.min(probability);
            }
        }
    }
    entities.sort_by_key(|e| (e.start(), e.end()));
    entities
}

impl crate::sealed::Sealed for AdaptNer {}

impl Model for AdaptNer {
    fn extract_entities(&self, text: &str, _: Option<Language>) -> Result<Vec<Entity>> {
        let mut ranges = Vec::new();
        let words: Vec<_> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let mut prev_end = 0;
        for &word in &words {
            let start = prev_end
                + text[prev_end..]
                    .find(word)
                    .ok_or_else(|| inference("source word alignment failed"))?;
            prev_end = start + word.len();
            ranges.push(start..prev_end);
        }
        Ok(decode(text, &ranges, &self.predict(&words)?))
    }
    fn supported_types(&self) -> Vec<EntityType> {
        LABELS.iter().map(|s| entity_type(s)).collect()
    }
    fn is_available(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "Arabic-AdaptNER"
    }
    fn version(&self) -> String {
        REVISION.to_owned()
    }
    fn description(&self) -> &'static str {
        "Arabic nested Wojood NER using independent per-type BIO channels (ONNX)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_channels_begin_precedence_and_orphan_inside() {
        let text = "حرب\t1948 1948";
        let mut logits = [[0.0; CHANNELS]; 3];
        // EVENT wraps a DATE. A second DATE opens even if I is positive.
        logits[0][21 + 3] = 1.0;
        logits[1][21 + 3] = 1.0;
        logits[1][2] = 2.0;
        logits[2][2] = 1.0;
        logits[2][21 + 2] = 3.0;
        let entities = decode(text, &[0..6, 7..11, 12..16], &logits);
        assert_eq!(
            entities
                .iter()
                .map(|e| (
                    e.entity_type.as_label(),
                    e.start(),
                    e.end(),
                    e.text.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("EVENT", 0, 8, "حرب\t1948"),
                ("DATE", 4, 8, "1948"),
                ("DATE", 9, 13, "1948")
            ]
        );
    }
    #[test]
    #[ignore = "requires the pinned local AdaptNER ONNX export in ADAPTNER_MODEL_DIR"]
    fn released_weights_recover_nested_date_in_event() {
        let model = AdaptNer::from_env().unwrap();
        let text =
            "خارطة فلسطين تبين طرق التهجير التي سلكها الفلسطينيون أثناء حرب 1948 ( النكبة ) .";
        let entities = model.extract_entities(text, None).unwrap();
        let has = |label: &str, surface: &str| {
            entities.iter().any(|e| {
                e.entity_type.as_label() == label
                    && e.text == surface
                    && text
                        .chars()
                        .skip(e.start())
                        .take(e.end() - e.start())
                        .collect::<String>()
                        == surface
            })
        };
        assert!(has("DATE", "1948"));
        assert!(has("EVENT", "حرب 1948"));
        assert!(has("GPE", "فلسطين"));
    }
}
