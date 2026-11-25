//! GLiNER2.5 zero-shot entity extraction from local FP32 boundary exports.
//!
//! Load the `fp32_25` directory from `jugaadsrl/gliner2.5-multi-v1-onnx`
//! (reference revision `3de2c11f55853b7919f985a59bfc8df27b5ec991`).
//! Its tokenizer, encoder and bucket-specific heads must be present, including
//! the heads' external ONNX data files. The boundary head returns word spans
//! and per-label logits. Anno preserves source character offsets and applies
//! the manifest's flat overlap policy separately for each label.

use super::gliner2_fastino::processor::{SchemaTask, SchemaTransformer};
use super::hf_loader::{create_onnx_session, load_tokenizer};
use super::inference::ZeroShotNER;
use crate::{Entity, EntityType, Error, Language, Model, Result};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path, sync::Mutex};

const DEFAULT_LABELS: &[&str] = &["person", "organization", "location", "date"];

#[derive(Deserialize)]
struct Manifest {
    architecture: String,
    hidden_size: usize,
    pool_size: usize,
    length_buckets: Vec<usize>,
    enable_abstention: bool,
    overlap_policy: String,
    token_pooling: String,
    max_position_embeddings: usize,
}

impl Manifest {
    fn load(dir: &Path) -> Result<(std::path::PathBuf, Self)> {
        let dir = if dir.join("fp32_25").is_dir() {
            dir.join("fp32_25")
        } else {
            dir.to_owned()
        };
        let m: Manifest =
            serde_json::from_slice(&std::fs::read(dir.join("boundary_manifest.json"))?)
                .map_err(|e| Error::ModelInit(format!("invalid GLiNER2.5 manifest: {e}")))?;
        if m.architecture != "boundary"
            || m.token_pooling != "first"
            || m.overlap_policy != "flat"
            || m.hidden_size == 0
            || m.pool_size == 0
            || m.max_position_embeddings == 0
            || m.length_buckets.is_empty()
            || m.length_buckets.contains(&0)
        {
            return Err(Error::ModelInit("unsupported GLiNER2.5 manifest: expected a first-token boundary export with flat overlap and positive dimensions".into()));
        }
        Ok((dir, m))
    }
}

fn boundary_head_filename(bucket: usize) -> String {
    format!("boundary_head_L{bucket}_fp32.onnx")
}

/// CPU FP32 GLiNER2.5 entity extractor for the Fastino multilingual boundary model.
pub struct GLiNER25 {
    directory: std::path::PathBuf,
    transformer: SchemaTransformer,
    manifest: Manifest,
    encoder: Mutex<Session>,
    heads: BTreeMap<usize, Mutex<Session>>,
}

fn inference(e: impl std::fmt::Display) -> Error {
    Error::Inference(format!("GLiNER2.5: {e}"))
}

impl GLiNER25 {
    /// Files used for evaluating this backend. External tensor data contributes to the result.
    pub fn artifact_paths(&self) -> Vec<(String, std::path::PathBuf)> {
        let mut files = vec![
            ("tokenizer".into(), self.directory.join("tokenizer.json")),
            (
                "manifest".into(),
                self.directory.join("boundary_manifest.json"),
            ),
        ];
        for stem in std::iter::once("encoder_fp32".to_owned()).chain(
            self.heads
                .keys()
                .map(|b| format!("boundary_head_L{b}_fp32")),
        ) {
            let graph = self.directory.join(format!("{stem}.onnx"));
            let data = graph.with_extension("onnx.data");
            if data.is_file() {
                files.push((format!("{stem}_data"), data));
            }
            files.push((stem, graph));
        }
        files
    }

    /// Load the export selected by the CLI and evaluation backend.
    pub fn from_env() -> Result<Self> {
        let dir = std::env::var_os("GLINER25_MODEL_DIR").ok_or_else(|| {
            Error::ModelInit("set GLINER25_MODEL_DIR to the local GLiNER2.5 FP32 export".into())
        })?;
        Self::from_local(Path::new(&dir))
    }

    /// Check the required files for automatic evaluation without opening ONNX sessions.
    /// ONNX Runtime validates the graph contents and any external tensor data at load time.
    pub fn model_files_available(dir: &Path) -> bool {
        Manifest::load(dir).is_ok_and(|(dir, manifest)| {
            dir.join("tokenizer.json").is_file()
                && dir.join("encoder_fp32.onnx").is_file()
                && manifest
                    .length_buckets
                    .iter()
                    .all(|&bucket| dir.join(boundary_head_filename(bucket)).is_file())
        })
    }

    /// Loads a directory containing the FP32 graph fragments and tokenizer.
    pub fn from_local(dir: &Path) -> Result<Self> {
        let (dir, m) = Manifest::load(dir)?;
        // Each head owns its own ORT session. A machine-sized thread pool
        // per bucket can exhaust the process thread budget before inference.
        let load = |name: &str| {
            let config = super::hf_loader::OnnxSessionConfig {
                num_threads: 2,
                ..Default::default()
            };
            create_onnx_session(&dir.join(name), config).map(Mutex::new)
        };
        let transformer = SchemaTransformer::new(load_tokenizer(&dir.join("tokenizer.json"))?)?;
        let encoder = load("encoder_fp32.onnx")?;
        let heads = m
            .length_buckets
            .iter()
            .map(|&n| Ok((n, load(&boundary_head_filename(n))?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            directory: dir,
            transformer,
            manifest: m,
            heads,
            encoder,
        })
    }
}

#[derive(Clone)]
struct Candidate {
    start: usize,
    end: usize,
    score: f32,
}

// The flat policy maximizes total score over disjoint half-open intervals.
// A single greedy longest/highest-scoring candidate can suppress a better pair.
fn flat(mut items: Vec<Candidate>) -> Vec<Candidate> {
    items.sort_by(|a, b| {
        a.end
            .cmp(&b.end)
            .then(a.start.cmp(&b.start))
            .then(b.score.total_cmp(&a.score))
    });
    items.dedup_by_key(|s| (s.start, s.end));
    let mut best = vec![(0.0f64, Vec::<usize>::new())];
    for (i, c) in items.iter().enumerate() {
        let prior = items[..i].partition_point(|p| p.end <= c.start);
        let mut take = best[prior].clone();
        take.0 += f64::from(c.score);
        take.1.push(i);
        let skip = &best[i];
        let ranks = |selection: &[usize]| {
            let mut rank = selection.to_vec();
            rank.sort_by(|&a, &b| {
                items[b]
                    .score
                    .total_cmp(&items[a].score)
                    .then(items[a].start.cmp(&items[b].start))
                    .then(items[a].end.cmp(&items[b].end))
            });
            rank.into_iter()
                .map(|j| {
                    (
                        std::cmp::Reverse(items[j].score.to_bits()),
                        items[j].start,
                        items[j].end,
                    )
                })
                .collect::<Vec<_>>()
        };
        let use_take = take.0 > skip.0
            || take.0 == skip.0
                && (take.1.len() > skip.1.len()
                    || take.1.len() == skip.1.len() && ranks(&take.1) < ranks(&skip.1));
        best.push(if use_take { take } else { skip.clone() });
    }
    best.pop()
        .unwrap_or_default()
        .1
        .into_iter()
        .map(|i| items[i].clone())
        .collect()
}

impl ZeroShotNER for GLiNER25 {
    fn default_types(&self) -> &[&'static str] {
        DEFAULT_LABELS
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        self.extract_with_types(text, labels, threshold)
    }

    fn extract_with_types(
        &self,
        text: &str,
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::InvalidInput(
                "expected a finite probability threshold in [0,1]".into(),
            ));
        }
        if text.trim().is_empty() || labels.is_empty() {
            return Ok(vec![]);
        }
        let task = SchemaTask::Entities(labels.iter().map(|s| s.to_string()).collect());
        let record = self.transformer.transform(text, &[task])?;
        let words = record.word_to_token_maps.len();
        let seq = record.input_ids.len();
        if seq == 0 {
            return Err(inference("empty tokenized prompt"));
        }
        if seq > self.manifest.max_position_embeddings {
            return Err(Error::InvalidInput(format!(
                "GLiNER2.5 prompt and text need {seq} tokens; maximum is {}",
                self.manifest.max_position_embeddings
            )));
        }
        let (&bucket, head) = self.heads.range(words..).next().ok_or_else(|| {
            Error::InvalidInput(format!("GLiNER2.5 has no boundary head for {words} words"))
        })?;
        let h = self.manifest.hidden_size;
        let q = labels.len();
        let input_ids = Tensor::from_array(([1, seq], record.input_ids)).map_err(inference)?;
        let mask = Tensor::from_array(([1, seq], record.attention_mask)).map_err(inference)?;
        // The exported routed_gather is first-token lookup with zero padding.
        let mut text_states = vec![
            0f32;
            bucket
                .checked_mul(h)
                .ok_or_else(|| inference("word tensor too large"))?
        ];
        let mut query_states = Vec::new();
        {
            let mut enc = self.encoder.lock().map_err(inference)?;
            let outputs = enc
                .run(ort::inputs!["input_ids" => input_ids, "attention_mask" => mask])
                .map_err(inference)?;
            let (shape, data) = outputs
                .get("last_hidden_state")
                .ok_or_else(|| inference("missing last_hidden_state"))?
                .try_extract_tensor::<f32>()
                .map_err(inference)?;
            if shape[..] != [1, seq as i64, h as i64] {
                return Err(inference("invalid encoder hidden-state shape"));
            }
            let get_row = |idx: usize| {
                let idx = idx.min(seq - 1);
                &data[idx * h..(idx + 1) * h]
            };
            for (row, &(first, _)) in text_states.chunks_mut(h).zip(&record.word_to_token_maps) {
                row.copy_from_slice(get_row(first));
            }
            for &idx in &record.tasks[0].field_tok_indices {
                query_states.extend_from_slice(get_row(idx));
            }
        }
        let mut text_mask = vec![0i64; bucket];
        text_mask[..words].fill(1);
        let mut head = head.lock().map_err(inference)?;
        let out = head.run(ort::inputs![
            "text_states" => Tensor::from_array(([1,bucket,h], text_states)).map_err(inference)?,
            "text_mask" => Tensor::from_array(([1,bucket], text_mask)).map_err(inference)?,
            "query_states" => Tensor::from_array(([1,q,h], query_states)).map_err(inference)?,
            "query_mask" => Tensor::from_array(([1,q], vec![1i64;q])).map_err(inference)?
        ]).map_err(inference)?;
        let c = self.manifest.pool_size;
        macro_rules! tensor {
            ($name:literal, $ty:ty, $shape:expr) => {{
                let (shape, data) = out
                    .get($name)
                    .ok_or_else(|| inference(concat!("missing ", $name)))?
                    .try_extract_tensor::<$ty>()
                    .map_err(inference)?;
                if shape[..] != $shape {
                    return Err(inference(concat!("invalid shape for ", $name)));
                }
                data
            }};
        }
        let idx = tensor!("cand_indices", i64, [1, q as i64, c as i64, 2]);
        let logits = tensor!("pair_logits", f32, [1, q as i64, c as i64]);
        let valid = tensor!("cand_valid", bool, [1, q as i64, c as i64]);
        let null = tensor!("null_logits", f32, [1, q as i64]);
        let mut result = Vec::new();
        let offsets = crate::SpanConverter::new(text);
        for (j, label) in labels.iter().enumerate() {
            let valid_rows = (j * c..(j + 1) * c).filter(|&k| valid[k]);
            if self.manifest.enable_abstention && valid_rows.clone().all(|k| logits[k] < null[j]) {
                continue;
            }
            let mut candidates = Vec::new();
            for k in valid_rows {
                let score = 1.0 / (1.0 + (-logits[k]).exp());
                let (start, end) = (idx[2 * k] as usize, idx[2 * k + 1] as usize);
                if score.is_finite()
                    && score >= threshold
                    && start < end
                    && end <= words
                    && record.word_to_char_maps[start].0 < record.word_to_char_maps[end - 1].1
                {
                    candidates.push(Candidate { start, end, score });
                }
            }
            for span in flat(candidates) {
                let begin = record.word_to_char_maps[span.start].0;
                let end = record.word_to_char_maps[span.end - 1].1;
                result.push(Entity::new(
                    &text[begin..end],
                    crate::schema::map_to_canonical(label, None),
                    offsets.byte_to_char(begin),
                    offsets.byte_to_char(end),
                    span.score,
                ));
            }
        }
        result.sort_by_key(|e| (e.start(), e.end()));
        Ok(result)
    }
}

impl Model for GLiNER25 {
    fn extract_entities(&self, text: &str, _: Option<Language>) -> Result<Vec<Entity>> {
        self.extract_with_types(text, DEFAULT_LABELS, 0.5)
    }
    fn supported_types(&self) -> Vec<EntityType> {
        DEFAULT_LABELS
            .iter()
            .map(|s| EntityType::from_label(s))
            .collect()
    }
    fn is_available(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "GLiNER2.5"
    }
    fn description(&self) -> &'static str {
        "Fastino GLiNER2.5 FP32 boundary NER (ONNX)"
    }
    fn as_zero_shot(&self) -> Option<&dyn ZeroShotNER> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_policy_chooses_two_adjacent_mentions_over_one() {
        let candidate = |start, end, score| Candidate { start, end, score };
        let spans = flat(vec![
            candidate(0, 4, 0.9),
            candidate(0, 2, 0.6),
            candidate(2, 4, 0.5),
            candidate(0, 2, 0.4),
        ]);
        assert_eq!(
            spans
                .into_iter()
                .map(|s| (s.start, s.end))
                .collect::<Vec<_>>(),
            [(0, 2), (2, 4)]
        );
    }

    #[test]
    fn selection_requires_every_manifest_bucket_in_the_selected_layout() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path();
        let manifest = br#"{"architecture":"boundary","token_pooling":"first","overlap_policy":"flat","hidden_size":4,"pool_size":1,"max_position_embeddings":256,"length_buckets":[32,64],"enable_abstention":false}"#;
        assert!(!GLiNER25::model_files_available(dir));
        std::fs::write(dir.join("boundary_manifest.json"), manifest).unwrap();
        for filename in [
            "tokenizer.json",
            "encoder_fp32.onnx",
            "boundary_head_L32_fp32.onnx",
        ] {
            std::fs::write(dir.join(filename), []).unwrap();
        }
        assert!(!GLiNER25::model_files_available(dir));
        std::fs::write(dir.join("boundary_head_L64_fp32.onnx"), []).unwrap();
        // Presence is sufficient here: weights may be inline and validation belongs to ORT.
        assert!(GLiNER25::model_files_available(dir));

        let preferred = dir.join("fp32_25");
        std::fs::create_dir(&preferred).unwrap();
        assert!(!GLiNER25::model_files_available(dir));
        for file in std::fs::read_dir(dir).unwrap() {
            let file = file.unwrap();
            if file.path().is_file() {
                std::fs::copy(file.path(), preferred.join(file.file_name())).unwrap();
            }
        }
        assert!(GLiNER25::model_files_available(dir));
        std::fs::write(preferred.join("boundary_manifest.json"), b"{}").unwrap();
        assert!(!GLiNER25::model_files_available(dir));
    }

    #[test]
    #[ignore = "requires GLINER25_MODEL_DIR and a local FP32 GLiNER2.5 multilingual export"]
    fn multilingual_checkpoint_preserves_unicode_offsets() {
        let model = GLiNER25::from_env().unwrap();
        let text = "Éva works for Acme.";
        let entities = model
            .extract_with_types(text, &["person", "organization"], 0.5)
            .unwrap();
        assert!(entities
            .iter()
            .any(|e| e.text == "Acme" && e.start() == 14 && e.end() == 18));
        assert!(entities
            .iter()
            .any(|e| e.text == "Éva" && e.start() == 0 && e.end() == 3));

        // gliner2 2.2.0, multilingual checkpoint 7b00527. Upstream adds a
        // period before encoding. Omitting it shifts this probability by .0074.
        let unpunctuated = "Mario Rossi works at Apple in Cupertino";
        let entities = model
            .extract_with_types(unpunctuated, &["person", "organization", "location"], 0.5)
            .unwrap();
        let city = entities.iter().find(|e| e.text == "Cupertino").unwrap();
        assert_eq!((city.start(), city.end()), (30, 39));
        assert!((city.confidence.value() - 0.9982).abs() < 0.001);
    }
}
