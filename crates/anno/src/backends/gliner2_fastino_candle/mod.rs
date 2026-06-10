//! # gliner2_fastino_candle (Phase 4)
//!
//! Candle backend for fastino-ai GLiNER2 with **runtime LoRA adapter
//! merge-at-load**. Loads PEFT-format adapters and merges them into the
//! base weights at `load_adapter` time, producing a fully-merged model
//! with zero per-forward overhead.
//!
//! Parallel to the ONNX-based [`crate::backends::gliner2_fastino`]. Same
//! public method shapes (Model + ZeroShotNER); users swap backends with
//! a type alias. The differentiator is `load_adapter` / `unload_adapter`.
//!
//! ## When to use this backend
//!
//! - You have multiple domain-specific LoRA adapters (e.g., legal,
//!   medical, financial) trained on the same base model.
//! - You want to switch between domains at runtime without re-exporting
//!   merged ONNX models per domain (which costs ~6 GB on disk per).
//! - Adapter swap rate is moderate (every few minutes/hours, not per
//!   request). For sub-millisecond hot-swap, see optional Phase 4.5.
//!
//! ## Architecture
//!
//! - Encoder: [`candle_transformers::models::debertav2::DebertaV2Model`]
//!   — provides DeBERTa-v2/v3 disentangled attention without anno
//!   reimplementing it ([PR #2743](https://github.com/huggingface/candle/pull/2743)).
//! - Heads: 7 small Candle modules (token_gather, span_rep, schema_gather,
//!   count_pred, count_lstm, scorer, classifier).
//! - LoRA: `W_merged = W_base + (alpha/r) * (lora_B @ lora_A)`, applied
//!   once at `load_adapter` time per target module.

#![cfg(feature = "gliner2-fastino-candle")]
#![allow(dead_code)] // Phase 4 in-progress: methods wired by M5+

pub mod decoder;
pub mod encoder;
pub mod heads;
pub mod lora;
pub mod pipeline;
pub mod processor;

use std::path::{Path, PathBuf};

use candle_core::Device;

/// Phase 4 Candle-based GLiNER2 backend with PEFT LoRA adapter
/// merge-at-load support.
pub struct GLiNER2FastinoCandle {
    pub(crate) tokenizer: tokenizers::Tokenizer,
    pub(crate) device: Device,
    /// Directory containing the base model's tokenizer.json,
    /// config.json, and model.safetensors. Used to re-merge from disk
    /// when `unload_adapter` is called or a new adapter replaces a
    /// previous one.
    pub(crate) base_model_dir: PathBuf,
    pub(crate) encoder: encoder::Encoder,
    pub(crate) heads: heads::AllHeads,
    /// Name of the currently merged adapter, or `None` if running on
    /// pure base weights.
    pub(crate) active_adapter: Option<String>,
    pub(crate) model_id: String,
}

impl std::fmt::Debug for GLiNER2FastinoCandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLiNER2FastinoCandle")
            .field("model_id", &self.model_id)
            .field("active_adapter", &self.active_adapter)
            .finish()
    }
}

impl GLiNER2FastinoCandle {
    /// Active adapter name, or `None` if running on pure base weights.
    pub fn active_adapter(&self) -> Option<&str> {
        self.active_adapter.as_deref()
    }

    /// Load a PEFT-format LoRA adapter and merge it into the base
    /// weights. Replaces any previously-active adapter (the engine
    /// reloads from the cached `base_model_dir` and re-applies the new
    /// delta).
    ///
    /// Cost: ~100 ms per call (safetensors re-read + per-target-module
    /// matmul + add). Subsequent inference is identical to running the
    /// merged model — zero per-forward overhead.
    ///
    /// Returns an error if:
    /// - `adapter_dir` is not a valid PEFT adapter (missing `adapter_config.json`
    ///   or `adapter_model.safetensors`/`adapter_weights.safetensors`).
    /// - The adapter targets modules that don't exist in the base.
    /// - The adapter's `base_model_name_or_path` doesn't match this engine's
    ///   `model_id` (defensive; prevents merging a "legal" adapter trained
    ///   on `gliner2-base-v1` into a `gliner2-multi-v1` base).
    pub fn load_adapter(&mut self, name: &str, adapter_dir: &Path) -> crate::Result<()> {
        let adapter = lora::LoraAdapter::load(adapter_dir, &self.device)?;

        // Defensive: reject mismatched base models if recorded in the adapter.
        if let Some(adapter_base) = adapter.config.base_model_name_or_path.as_deref() {
            if !self.model_id.contains(adapter_base) && !adapter_base.contains(&self.model_id) {
                return Err(crate::Error::Backend(format!(
                    "load_adapter: adapter trained on '{adapter_base}', current \
                     model is '{}'. Refusing to merge — remove \
                     base_model_name_or_path from adapter_config.json to bypass.",
                    self.model_id
                )));
            }
        }

        // Merge the adapter into the base safetensors → in-memory tensor map.
        let base_safetensors = self.base_model_dir.join("model.safetensors");
        let merged = lora::merge_into_base(&base_safetensors, &adapter, &self.device)?;

        // Build a VarBuilder from the merged tensor map and rebuild encoder + heads.
        let vb = candle_nn::VarBuilder::from_tensors(merged, candle_core::DType::F32, &self.device);
        let new_encoder =
            encoder::Encoder::from_var_builder(vb.pp("encoder"), &self.encoder.config)?;
        let new_heads = heads::AllHeads::from_var_builder(vb, &self.device)?;

        self.encoder = new_encoder;
        self.heads = new_heads;
        self.active_adapter = Some(name.to_string());
        Ok(())
    }

    /// Discard the active adapter and reload pure base weights from
    /// `base_model_dir`. Idempotent — calling on an engine without an
    /// active adapter is a no-op.
    pub fn unload_adapter(&mut self) -> crate::Result<()> {
        if self.active_adapter.is_none() {
            return Ok(());
        }
        let weights_path = self.base_model_dir.join("model.safetensors");
        let config_path = if self
            .base_model_dir
            .join("encoder_config/config.json")
            .exists()
        {
            self.base_model_dir.join("encoder_config/config.json")
        } else {
            self.base_model_dir.join("config.json")
        };
        self.encoder =
            encoder::Encoder::from_safetensors(&weights_path, &config_path, &self.device)?;
        self.heads = heads::AllHeads::from_safetensors(&weights_path, &self.device)?;
        self.active_adapter = None;
        Ok(())
    }

    /// Load from a local directory containing the PyTorch artifacts:
    /// `tokenizer.json`, `config.json`, `model.safetensors`.
    ///
    /// **Phase 4 / experimental.** No GPU device override yet — uses
    /// CPU. Phase 4.5 may add `from_local_with_config` analogous to
    /// the ONNX backend's pattern.
    pub fn from_local(model_dir: &Path) -> crate::Result<Self> {
        Self::from_local_on_device(model_dir, &Device::Cpu)
    }

    /// Load from a local directory with an explicit Candle device.
    pub fn from_local_with_device(model_dir: &Path, device: &Device) -> crate::Result<Self> {
        Self::from_local_on_device(model_dir, device)
    }

    /// Load from HuggingFace Hub. Downloads `tokenizer.json`,
    /// `config.json`, and `model.safetensors` to the local HF cache,
    /// then defers to [`Self::from_local`].
    ///
    /// **Important**: this loads the *PyTorch* repo (e.g.
    /// `fastino/gliner2-multi-v1`), NOT the SemplificaAI ONNX export.
    /// They're different artifacts.
    pub fn from_pretrained(model_id: &str) -> crate::Result<Self> {
        Self::from_pretrained_with_device(model_id, &Device::Cpu)
    }

    /// Load from HuggingFace Hub with an explicit Candle device.
    pub fn from_pretrained_with_device(model_id: &str, device: &Device) -> crate::Result<Self> {
        let api = crate::backends::hf_loader::hf_api()
            .map_err(|e| crate::Error::Backend(format!("hf_api: {e}")))?;
        let repo = api.model(model_id.to_string());

        // Touch each required file so it's in the local cache. Order
        // matters: weights last so we can use its parent as the snapshot dir.
        let _tokenizer =
            crate::backends::hf_loader::download_model_file(&repo, &["tokenizer.json"])
                .map_err(|e| crate::Error::Backend(format!("download tokenizer: {e}")))?;
        let _config = crate::backends::hf_loader::download_model_file(&repo, &["config.json"])
            .map_err(|e| crate::Error::Backend(format!("download config: {e}")))?;
        // GLiNER2 stores the encoder's HF config (with vocab_size,
        // hidden_size, etc.) under encoder_config/config.json. The
        // top-level config.json is the GLiNER2 wrapper config.
        let _encoder_config =
            crate::backends::hf_loader::download_model_file(&repo, &["encoder_config/config.json"])
                .map_err(|e| crate::Error::Backend(format!("download encoder_config: {e}")))?;
        let weights_path = crate::backends::hf_loader::download_model_file(
            &repo,
            &["model.safetensors", "pytorch_model.bin"],
        )
        .map_err(|e| crate::Error::Backend(format!("download weights: {e}")))?;

        let snapshot_dir = weights_path
            .parent()
            .ok_or_else(|| crate::Error::Backend("snapshot dir resolution".into()))?;
        let mut model = Self::from_local_with_device(snapshot_dir, device)?;
        model.model_id = model_id.to_string();
        Ok(model)
    }

    /// Internal: like [`Self::from_local`] but explicit about the
    /// Candle device. Hot-swap on a different device requires re-
    /// loading; not exposed publicly until Phase 4.5.
    pub(crate) fn from_local_on_device(model_dir: &Path, device: &Device) -> crate::Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weights_path = model_dir.join("model.safetensors");
        // The encoder's HF config is under encoder_config/config.json
        // for GLiNER2 repos (top-level config.json is the wrapper).
        // Fall back to top-level config.json for direct DeBERTa snapshots.
        let nested_encoder_config = model_dir.join("encoder_config").join("config.json");
        let config_path = if nested_encoder_config.exists() {
            nested_encoder_config
        } else {
            model_dir.join("config.json")
        };

        if !weights_path.exists() {
            return Err(crate::Error::Backend(format!(
                "gliner2_fastino_candle: model.safetensors not found in {} \
                 (PyTorch fastino/gliner2-* repo expected; SemplificaAI ONNX \
                 export is a different artifact)",
                model_dir.display()
            )));
        }

        let tokenizer = crate::backends::hf_loader::load_tokenizer(&tokenizer_path)
            .map_err(|e| crate::Error::Backend(format!("tokenizer: {e}")))?;
        let encoder = encoder::Encoder::from_safetensors(&weights_path, &config_path, device)?;
        let heads = heads::AllHeads::from_safetensors(&weights_path, device)?;
        let model_id = model_dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "gliner2_fastino_candle_local".to_string());

        Ok(Self {
            tokenizer,
            device: device.clone(),
            base_model_dir: model_dir.to_path_buf(),
            encoder,
            heads,
            active_adapter: None,
            model_id,
        })
    }

    pub(crate) fn extract_ner(
        &self,
        text: &str,
        types: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<crate::Entity>> {
        if types.is_empty() {
            return Ok(vec![]);
        }
        let labels: Vec<String> = types.iter().map(|s| s.to_string()).collect();
        let task = processor::SchemaTask::Entities(labels);
        let transformer =
            processor::SchemaTransformer::new(self.tokenizer.clone()).map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: transformer: {e}"))
            })?;
        let record = transformer.transform(text, &[task]).map_err(|e| {
            crate::Error::Backend(format!("gliner2_fastino_candle: transform: {e}"))
        })?;
        let num_words = record.word_to_char_maps.len();
        if num_words == 0 {
            return Ok(vec![]);
        }
        let task_map = record.tasks.first().ok_or_else(|| {
            crate::Error::Backend(
                "gliner2_fastino_candle: transformer produced no task mapping".into(),
            )
        })?;

        let (scorer_out, pred_count) = pipeline::run_pipeline_candle(self, &record, task_map)?;
        if pred_count == 0 {
            return Ok(vec![]);
        }
        let entities = decoder::decode_entities(
            text,
            &record,
            task_map,
            &scorer_out,
            pred_count,
            threshold,
            /* flat_ner = */ false,
        );
        Ok(entities)
    }

    /// Extract entities using per-label descriptions in the prompt.
    ///
    /// Mirrors
    /// [`crate::backends::gliner2_fastino::GLiNER2Fastino::extract_with_label_descriptions`].
    pub fn extract_with_label_descriptions(
        &self,
        text: &str,
        labeled: &[(&str, &str)],
        threshold: f32,
    ) -> crate::Result<Vec<crate::Entity>> {
        if labeled.is_empty() {
            return Ok(vec![]);
        }
        let owned: Vec<(String, String)> = labeled
            .iter()
            .map(|(l, d)| (l.to_string(), d.to_string()))
            .collect();
        let task = processor::SchemaTask::EntitiesDescribed(owned);
        let transformer =
            processor::SchemaTransformer::new(self.tokenizer.clone()).map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: transformer: {e}"))
            })?;
        let record = transformer.transform(text, &[task]).map_err(|e| {
            crate::Error::Backend(format!("gliner2_fastino_candle: transform: {e}"))
        })?;
        let num_words = record.word_to_char_maps.len();
        if num_words == 0 {
            return Ok(vec![]);
        }
        let task_map = record.tasks.first().ok_or_else(|| {
            crate::Error::Backend(
                "gliner2_fastino_candle: transformer produced no task mapping".into(),
            )
        })?;

        let (scorer_out, pred_count) = pipeline::run_pipeline_candle(self, &record, task_map)?;
        if pred_count == 0 {
            return Ok(vec![]);
        }
        Ok(decoder::decode_entities(
            text,
            &record,
            task_map,
            &scorer_out,
            pred_count,
            threshold,
            /* flat_ner = */ false,
        ))
    }

    /// Extract entities with per-label thresholds.
    ///
    /// Mirrors
    /// [`crate::backends::gliner2_fastino::GLiNER2Fastino::extract_with_label_thresholds`].
    pub fn extract_with_label_thresholds(
        &self,
        text: &str,
        label_thresholds: &[(&str, f32)],
    ) -> crate::Result<Vec<crate::Entity>> {
        if label_thresholds.is_empty() {
            return Ok(vec![]);
        }
        let labels: Vec<String> = label_thresholds
            .iter()
            .map(|(l, _)| l.to_string())
            .collect();
        let task = processor::SchemaTask::Entities(labels);
        let transformer =
            processor::SchemaTransformer::new(self.tokenizer.clone()).map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: transformer: {e}"))
            })?;
        let record = transformer.transform(text, &[task]).map_err(|e| {
            crate::Error::Backend(format!("gliner2_fastino_candle: transform: {e}"))
        })?;
        let num_words = record.word_to_char_maps.len();
        if num_words == 0 {
            return Ok(vec![]);
        }
        let task_map = record.tasks.first().ok_or_else(|| {
            crate::Error::Backend(
                "gliner2_fastino_candle: transformer produced no task mapping".into(),
            )
        })?;

        let (scorer_out, pred_count) = pipeline::run_pipeline_candle(self, &record, task_map)?;
        if pred_count == 0 {
            return Ok(vec![]);
        }
        Ok(decoder::decode_entities_with_thresholds(
            text,
            &record,
            task_map,
            &scorer_out,
            pred_count,
            label_thresholds,
            /* flat_ner = */ false,
        ))
    }

    /// Extract structured data per the given schema.
    ///
    /// Mirrors
    /// [`crate::backends::gliner2_fastino::GLiNER2Fastino::extract_structure`].
    pub fn extract_structure(
        &self,
        text: &str,
        schema: &crate::backends::gliner2_fastino::schema::TaskSchema,
        threshold: f32,
    ) -> crate::Result<Vec<crate::backends::gliner2_fastino::schema::ExtractedStructure>> {
        if schema.structures.is_empty() {
            return Ok(vec![]);
        }
        let transformer =
            processor::SchemaTransformer::new(self.tokenizer.clone()).map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: transformer: {e}"))
            })?;

        let mut all_results: Vec<crate::backends::gliner2_fastino::schema::ExtractedStructure> =
            Vec::new();
        for st in &schema.structures {
            if st.fields.is_empty() {
                continue;
            }
            let fields_owned: Vec<(String, crate::backends::gliner2_fastino::schema::FieldType)> =
                st.fields
                    .iter()
                    .map(|f| (f.name.clone(), f.field_type))
                    .collect();
            let task = processor::SchemaTask::Structures(st.name.clone(), fields_owned.clone());
            let record = transformer.transform(text, &[task]).map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: transform: {e}"))
            })?;
            let num_words = record.word_to_char_maps.len();
            if num_words == 0 {
                continue;
            }
            let task_map = record.tasks.first().ok_or_else(|| {
                crate::Error::Backend(
                    "gliner2_fastino_candle: transformer produced no task mapping".into(),
                )
            })?;

            let (scorer_out, pred_count) = pipeline::run_pipeline_candle(self, &record, task_map)?;
            if pred_count == 0 {
                continue;
            }
            let task_results = decoder::decode_structure(
                text,
                &record,
                task_map,
                &scorer_out,
                pred_count,
                threshold,
                &fields_owned,
            );
            all_results.extend(task_results);
        }
        Ok(all_results)
    }

    /// Single-label classification using the dedicated `[L]`-head classifier.
    ///
    /// Mirrors [`crate::backends::gliner2_fastino::GLiNER2Fastino::classify`].
    pub fn classify(
        &self,
        text: &str,
        labels: &[&str],
        _threshold: f32,
    ) -> crate::Result<Vec<(String, f32)>> {
        if labels.is_empty() {
            return Ok(vec![]);
        }
        let label_strings: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
        let task = processor::SchemaTask::Classifications(
            "classification".to_string(),
            label_strings.clone(),
        );
        let transformer =
            processor::SchemaTransformer::new(self.tokenizer.clone()).map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: transformer: {e}"))
            })?;
        let record = transformer.transform(text, &[task]).map_err(|e| {
            crate::Error::Backend(format!("gliner2_fastino_candle: transform: {e}"))
        })?;
        let task_map = record.tasks.first().ok_or_else(|| {
            crate::Error::Backend(
                "gliner2_fastino_candle: transformer produced no task mapping".into(),
            )
        })?;

        let probs = pipeline::run_classify_pipeline_candle(self, &record, task_map)?;

        let mut out: Vec<(String, f32)> =
            label_strings.into_iter().zip(probs.into_iter()).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }
}

use crate::backends::inference::ZeroShotNER;
use crate::{EntityCategory, EntityType, Language};

impl crate::Model for GLiNER2FastinoCandle {
    fn extract_entities(
        &self,
        text: &str,
        _language: Option<Language>,
    ) -> crate::Result<Vec<crate::Entity>> {
        self.extract_ner(text, &["person", "organization", "location", "date"], 0.5)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::custom("misc", EntityCategory::Misc),
        ]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "GLiNER2FastinoCandle"
    }

    fn description(&self) -> &'static str {
        "fastino-ai GLiNER2 (NER + classification, Candle, runtime LoRA)"
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            zero_shot: true,
            ..Default::default()
        }
    }

    fn as_zero_shot(&self) -> Option<&dyn ZeroShotNER> {
        Some(self)
    }
}

impl ZeroShotNER for GLiNER2FastinoCandle {
    fn default_types(&self) -> &[&'static str] {
        &["person", "organization", "location", "date", "event"]
    }

    fn extract_with_types(
        &self,
        text: &str,
        types: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<crate::Entity>> {
        self.extract_ner(text, types, threshold)
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<crate::Entity>> {
        self.extract_ner(text, descriptions, threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_local_with_device_rejects_missing_weights() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = GLiNER2FastinoCandle::from_local_with_device(dir.path(), &Device::Cpu)
            .expect_err("empty dir must fail");
        assert!(err.to_string().contains("model.safetensors"));
    }
}
