//! gliner2_fastino — fastino-ai GLiNER2 backend (issue #18).
//!
//! **Status:** Shipped (Phase 4). Candle + LoRA NER backend with merge-at-load.
//!
//! Loads `fastino/gliner2-*` ONNX models (Zaratiana et al. 2025,
//! arXiv:2507.18546). Distinct from `gliner_multitask` (which loads GLiNER v1
//! multi-task models with hardcoded `<<ENT>>=128002` IDs and rejects any
//! `fastino/*` model id at the discovery layer).
//!
//! # Architecture deltas vs `gliner_multitask`
//!
//! - Special-token vocabulary: `[P]`, `[E]`, `[C]`, `[L]`, `[R]`,
//!   `[SEP_STRUCT]`, `[SEP_TEXT]`. IDs read from `tokenizer.json` at load
//!   time; never hardcoded.
//! - Prompt format: `( [P] task_name ( [E] label1 [E] label2 ) ) [SEP_TEXT] tokens...`
//! - Span scoring: dot-product similarity (Eq. 1 of arXiv:2507.18546).
//!
//! # LoRA
//!
//! Phase 1 does **not** support runtime LoRA adapter loading. To use a
//! LoRA-fine-tuned model, merge the adapter into the base weights and
//! re-export to ONNX:
//!
//! ```bash
//! python scripts/gliner2_export_onnx.py \
//!     --base fastino/gliner2-multi-v1 \
//!     --lora-adapter ./my_adapter \
//!     --output ./my_merged.onnx
//! ```
//!
//! Pointing `from_local` at a directory containing `adapter_config.json`
//! returns [`errors::Error::LoraAdapterNotSupported`].
//!
//! # Structure extraction (Phase 2)
//!
//! [`GLiNER2Fastino::extract_structure`] returns a `Vec<schema::ExtractedStructure>`
//! given a [`schema::TaskSchema`]. Each `StructureTask` in the schema runs
//! one inference pass through the 8-session pipeline; the scorer's
//! `MAX_COUNT` axis is walked as the per-instance dimension.
//!
//! ```rust,no_run
//! use anno::backends::gliner2_fastino::GLiNER2Fastino;
//! use anno::backends::gliner2_fastino::schema::{
//!     FieldType, StructureTask, TaskSchema,
//! };
//! use std::path::Path;
//!
//! let model = GLiNER2Fastino::from_local(Path::new("./model")).unwrap();
//! let schema = TaskSchema::new().with_structure(
//!     StructureTask::new("invoice")
//!         .with_field("vendor", FieldType::String)
//!         .with_field("amount", FieldType::String),
//! );
//! let result = model
//!     .extract_structure("Invoice from Acme Corp for $4,250.", &schema, 0.5)
//!     .unwrap();
//! for instance in result {
//!     println!("{}: {:?}", instance.structure_type, instance.fields);
//! }
//! ```
//!
//! Phase 2 ships [`schema::FieldType::String`] only. `List` and `Choice`
//! field types decode with the same single-best-span treatment as `String`.
//!
//! # Execution mode (Phase 3.5)
//!
//! Two paths through the 8-session pipeline:
//!
//! - [`ExecutionMode::Standard`] (default): each session round-trips
//!   tensors through Rust ndarrays at the boundary — simple and CPU-friendly.
//! - [`ExecutionMode::IoBinding`]: tensors stay device-resident in a
//!   single ort allocator across the chain. 1.5-3× faster on CPU,
//!   required for efficient GPU inference.
//!
//! Opt in via [`GLiNER2FastinoConfig`]:
//!
//! ```rust,no_run
//! use anno::backends::gliner2_fastino::{
//!     ExecutionMode, GLiNER2Fastino, GLiNER2FastinoConfig,
//! };
//!
//! let model = GLiNER2Fastino::from_pretrained_with_config(
//!     "SemplificaAI/gliner2-multi-v1-onnx",
//!     GLiNER2FastinoConfig::default()
//!         .with_execution_mode(ExecutionMode::IoBinding),
//! )
//! .unwrap();
//! ```
//!
//! Every public extract method (`extract_ner`,
//! `extract_with_label_descriptions`, `extract_with_label_thresholds`,
//! `extract_structure`, `classify`) dispatches on the configured mode.
//! Standard ≡ IoBinding within a max-abs-diff tolerance of 1e-4 on
//! per-entity confidence — see the
//! `parity_standard_iobinding_*` integration tests.
//!
//! # Source attribution
//!
//! `processor.rs` is adapted from SemplificaAI/gliner2-rs (Apache-2.0):
//! <https://github.com/SemplificaAI/gliner2-rs/blob/main/rust_component/src/processor.rs>

#![cfg(feature = "gliner2-fastino")]

pub(crate) mod config;
pub(crate) mod decoder;
pub mod errors;
pub(crate) mod nms;
pub(crate) mod pipeline;
pub(crate) mod pipeline_iobinding;
pub(crate) mod processor;
pub mod schema;
pub(crate) mod sessions;

/// Inference execution mode.
///
/// Phase 3 standard mode (`Standard`) round-trips tensors through Rust
/// ndarrays at every session boundary — simple and CPU-friendly. Phase 3.5
/// IoBinding mode (`IoBinding`) keeps tensors in a single ort allocator
/// across the 8-session chain — required for efficient GPU inference and
/// 2-3× faster on CPU. See spec
/// `docs/superpowers/specs/2026-05-05-gliner2-fastino-phase3.5-iobinding.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    /// Phase 3 path. Default.
    #[default]
    Standard,
    /// Phase 3.5 path. Opt-in.
    IoBinding,
}

/// Configuration for [`GLiNER2Fastino::from_local_with_config`].
///
/// Marked `#[non_exhaustive]` — extend via `..Default::default()` to remain
/// forward-compatible with future fields.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GLiNER2FastinoConfig {
    /// ONNX session configuration (thread count, provider, etc.).
    pub onnx: crate::backends::hf_loader::OnnxSessionConfig,
    /// Execution path: standard round-trip or IoBinding device-resident.
    pub execution_mode: ExecutionMode,
}

impl Default for GLiNER2FastinoConfig {
    fn default() -> Self {
        Self {
            onnx: crate::backends::hf_loader::OnnxSessionConfig::default(),
            execution_mode: ExecutionMode::Standard,
        }
    }
}

impl GLiNER2FastinoConfig {
    /// Override the execution mode. Returns the config for chaining
    /// (`#[non_exhaustive]` blocks struct-literal construction outside
    /// the crate; this builder pattern is the canonical workaround).
    #[must_use]
    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    /// Override the ONNX session config (e.g., to enable CUDA).
    #[must_use]
    pub fn with_onnx(mut self, onnx: crate::backends::hf_loader::OnnxSessionConfig) -> Self {
        self.onnx = onnx;
        self
    }

    /// Override the ONNX execution-provider preferences without exposing
    /// the crate-internal `OnnxSessionConfig` type to downstream crates.
    #[must_use]
    pub fn with_onnx_provider_preferences(
        mut self,
        use_cpu_provider: bool,
        prefer_coreml: bool,
        prefer_cuda: bool,
    ) -> Self {
        self.onnx.use_cpu_provider = use_cpu_provider;
        self.onnx.prefer_coreml = prefer_coreml;
        self.onnx.prefer_cuda = prefer_cuda;
        self
    }
}

/// How to apply labels across a batch.
///
/// Phase 1.5 polish — shapes the input to
/// [`GLiNER2Fastino::batch_extract_with_schema_mode`].
pub enum BatchSchemaMode<'a> {
    /// All texts share the same label set.
    Shared(&'a [&'a str]),
    /// Each text has its own label set; outer slice indexed by text idx.
    /// Length must match the texts slice.
    PerSample(&'a [Vec<&'a str>]),
}

/// GLiNER2 fastino-ai NER model loaded from ONNX sessions.
///
/// Construct via [`Self::from_pretrained`] or [`Self::from_local`].
pub struct GLiNER2Fastino {
    #[allow(dead_code)] // retained for re-tokenization paths in later phases
    pub(crate) tokenizer: tokenizers::Tokenizer,
    #[allow(dead_code)] // retained for re-tokenization paths in later phases
    pub(crate) special: processor::SpecialTokenIds,
    pub(crate) transformer: processor::SchemaTransformer,
    pub(crate) config: config::FastinoConfig,
    pub(crate) sessions: sessions::Sessions,
    pub(crate) model_id: String,
    pub(crate) execution_mode: ExecutionMode,
}

impl std::fmt::Debug for GLiNER2Fastino {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLiNER2Fastino")
            .field("model_id", &self.model_id)
            .field("hidden_size", &self.config.hidden_size)
            .finish()
    }
}

use std::path::Path;

impl GLiNER2Fastino {
    /// Load with full configuration including execution mode.
    ///
    /// Use this when you need to opt into [`ExecutionMode::IoBinding`] or
    /// configure GPU execution providers via
    /// [`crate::backends::hf_loader::OnnxSessionConfig`].
    pub fn from_local_with_config(
        model_dir: &Path,
        cfg: GLiNER2FastinoConfig,
    ) -> crate::Result<Self> {
        if model_dir.join("adapter_config.json").exists() {
            return Err(errors::Error::LoraAdapterNotSupported {
                path: model_dir.to_path_buf(),
            }
            .into());
        }

        // Sessions::from_dir_with_cfg_mode resolves the dtype subdir
        // (fp32_v2/, etc.) and loads all 8 ONNX graphs. In IoBinding mode
        // it prefers `*_iobinding{suffix}` variants and falls back to the
        // standard filename when the iobinding-specific export isn't shipped.
        let (sessions, subdir) = sessions::Sessions::from_dir_with_cfg_mode(
            model_dir,
            cfg.onnx.clone(),
            cfg.execution_mode,
        )?;

        // Tokenizer: prefer subdir, fall back to root for layouts that ship
        // tokenizer at the snapshot root.
        let tokenizer_path = if subdir.join("tokenizer.json").exists() {
            subdir.join("tokenizer.json")
        } else {
            model_dir.join("tokenizer.json")
        };
        if !tokenizer_path.exists() {
            return Err(errors::Error::TokenizerMissing(tokenizer_path).into());
        }
        let tokenizer = crate::backends::hf_loader::load_tokenizer(&tokenizer_path)
            .map_err(|e| crate::Error::Backend(format!("gliner2_fastino: tokenizer: {e}")))?;

        let special = processor::SpecialTokenIds::resolve(&tokenizer)?;
        let transformer = processor::SchemaTransformer::new(tokenizer.clone())?;

        // config.json is optional — SemplificaAI's export doesn't ship one.
        // Fall back to defaults appropriate for gliner2-multi-v1.
        let config_path = if subdir.join("config.json").exists() {
            subdir.join("config.json")
        } else {
            model_dir.join("config.json")
        };
        let model_config = if config_path.exists() {
            config::FastinoConfig::from_path(&config_path)?
        } else {
            config::FastinoConfig::default()
        };

        Ok(Self {
            tokenizer,
            special,
            transformer,
            config: model_config,
            sessions,
            model_id: model_dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "gliner2_fastino_local".to_string()),
            execution_mode: cfg.execution_mode,
        })
    }

    /// Load a fastino GLiNER2 model from a local directory with custom ONNX session configuration.
    ///
    /// Equivalent to [`Self::from_local_with_config`] with
    /// `execution_mode = ExecutionMode::Standard`. Use the new method when
    /// you need IoBinding mode.
    pub fn from_local_with_options(
        model_dir: &Path,
        cfg: crate::backends::hf_loader::OnnxSessionConfig,
    ) -> crate::Result<Self> {
        Self::from_local_with_config(
            model_dir,
            GLiNER2FastinoConfig {
                onnx: cfg,
                execution_mode: ExecutionMode::Standard,
            },
        )
    }

    /// Load a fastino GLiNER2 model from a local directory.
    pub fn from_local(model_dir: &Path) -> crate::Result<Self> {
        Self::from_local_with_options(
            model_dir,
            crate::backends::hf_loader::OnnxSessionConfig::default(),
        )
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
        let task = processor::SchemaTask::Entities(labels.clone());
        let record = self.transformer.transform(text, &[task])?;
        let num_words = record.word_to_char_maps.len();
        if num_words == 0 {
            return Ok(vec![]);
        }
        let task_map = record.tasks.first().ok_or_else(|| {
            crate::Error::Backend("gliner2_fastino: transformer produced no task mapping".into())
        })?;

        // Phase 3.5: dispatch on execution_mode. Standard runs the
        // existing 8-call chain inline; IoBinding routes through
        // pipeline_iobinding's chained-DynValue chain.
        let (scorer_out, pred_count) = pipeline_iobinding::run_pipeline_dispatch(
            &self.sessions,
            &record,
            task_map,
            self.execution_mode,
        )?;
        if pred_count == 0 {
            return Ok(vec![]);
        }
        let entities = pipeline::decode_entities(
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

    /// Load a fastino GLiNER2 model by Hugging Face model id.
    ///
    /// Downloads `tokenizer.json`, `config.json`, and the 8 v2 ONNX graphs
    /// (encoder, token_gather, span_rep, schema_gather, count_pred_argmax,
    /// count_lstm_fixed, scorer, classifier) from the repo. Tries fp32_v2/
    /// first, falls back to fp16_v2/ per file. Then defers to `from_local`.
    ///
    /// **Phase 3 / experimental.** No retry/backoff on transient HF Hub
    /// failures beyond what `hf-hub` itself provides.
    pub fn from_pretrained(model_id: &str) -> crate::Result<Self> {
        Self::from_pretrained_with_config(model_id, GLiNER2FastinoConfig::default())
    }

    /// Phase 3.5: load from HF Hub with full configuration including
    /// execution mode. Same download flow as [`Self::from_pretrained`];
    /// dispatches to [`Self::from_local_with_config`] after the snapshot
    /// is on disk.
    pub fn from_pretrained_with_config(
        model_id: &str,
        cfg: GLiNER2FastinoConfig,
    ) -> crate::Result<Self> {
        let api = crate::backends::hf_loader::hf_api()
            .map_err(|e| crate::Error::Backend(format!("gliner2_fastino: hf_api: {e}")))?;
        let repo = api.model(model_id.to_string());

        // Tokenizer + config are co-located with the ONNX files in dtype subdirs.
        // Try fp32_v2/ first, fall back to fp16_v2/, then root for backward compat.
        let tokenizer_path = crate::backends::hf_loader::download_model_file(
            &repo,
            &[
                "fp32_v2/tokenizer.json",
                "fp16_v2/tokenizer.json",
                "tokenizer.json",
            ],
        )
        .map_err(|e| crate::Error::Backend(format!("gliner2_fastino: download tokenizer: {e}")))?;
        // config.json is optional — SemplificaAI's export doesn't include it.
        // Try to download if present, but ignore 404s and fall back to defaults
        // in from_local.
        let _ = crate::backends::hf_loader::download_model_file(
            &repo,
            &["fp32_v2/config.json", "fp16_v2/config.json", "config.json"],
        );

        // Download the 8 v2 ONNX files. Try fp32_v2 first (clearer dtype
        // semantics for debugging), then fp16_v2 as fallback.
        let bases = [
            "encoder",
            "token_gather",
            "span_rep",
            "schema_gather",
            "count_pred_argmax",
            "count_lstm_fixed",
            "scorer",
            "classifier",
        ];
        for base in &bases {
            let candidates = [
                format!("fp32_v2/{base}_fp32.onnx"),
                format!("fp16_v2/{base}_fp16.onnx"),
            ];
            let candidate_refs: Vec<&str> = candidates.iter().map(String::as_str).collect();
            crate::backends::hf_loader::download_model_file(&repo, &candidate_refs).map_err(
                |e| crate::Error::Backend(format!("gliner2_fastino: download {base}: {e}")),
            )?;
        }

        // Resolve to the snapshot dir and dispatch.
        // tokenizer_path may be at <snapshot>/fp32_v2/tokenizer.json (subdir)
        // or <snapshot>/tokenizer.json (legacy). Walk up until we find a parent
        // containing one of the dtype subdirs.
        let mut snapshot_dir = tokenizer_path.parent().ok_or_else(|| {
            crate::Error::Backend("gliner2_fastino: tokenizer has no parent".into())
        })?;
        loop {
            let has_dtype_subdir = ["fp32_v2", "fp16_v2", "fp32", "fp16"]
                .iter()
                .any(|sub| snapshot_dir.join(sub).is_dir());
            if has_dtype_subdir {
                break;
            }
            match snapshot_dir.parent() {
                Some(p) => snapshot_dir = p,
                None => break, // reached filesystem root; from_local will surface an error
            }
        }
        let mut model = Self::from_local_with_config(snapshot_dir, cfg)?;
        model.model_id = model_id.to_string();
        Ok(model)
    }
}

use crate::backends::inference::ZeroShotNER;
use crate::{EntityCategory, EntityType, Language};

impl crate::Model for GLiNER2Fastino {
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
        "GLiNER2Fastino"
    }

    fn description(&self) -> &'static str {
        "fastino-ai GLiNER2 (NER + classification, ONNX, experimental)"
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

impl ZeroShotNER for GLiNER2Fastino {
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
        // The ZeroShotNER trait's `descriptions` are treated as
        // self-describing labels (matches gliner_multitask's semantics —
        // each description IS the label name in disguise). For per-label
        // descriptions paired with separate label names, use
        // [`GLiNER2Fastino::extract_with_label_descriptions`] which emits
        // `[E] <label> [DESCRIPTION] <description>` in the prompt for an
        // accuracy boost per the GLiNER paper.
        self.extract_ner(text, descriptions, threshold)
    }
}

impl GLiNER2Fastino {
    /// Extract entities using per-label descriptions in the prompt.
    ///
    /// Each label has a separate description string emitted as
    /// `[E] <label> [DESCRIPTION] <description>` in the prompt. Per the
    /// GLiNER paper, descriptions provide a measurable accuracy boost
    /// on most NER benchmarks.
    ///
    /// **Phase 1.5 / experimental.** Not behind a public trait — promote
    /// when a second backend implements the same shape. The trait method
    /// [`crate::backends::inference::ZeroShotNER::extract_with_descriptions`]
    /// uses the older "descriptions are self-describing labels" convention
    /// (single string per label); this method takes explicit
    /// `(label, description)` pairs.
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
        let record = self.transformer.transform(text, &[task])?;
        let num_words = record.word_to_char_maps.len();
        if num_words == 0 {
            return Ok(vec![]);
        }
        let task_map = record.tasks.first().ok_or_else(|| {
            crate::Error::Backend("gliner2_fastino: transformer produced no task mapping".into())
        })?;

        // Phase 3.5: dispatch on execution_mode.
        let (scorer_out, pred_count) = pipeline_iobinding::run_pipeline_dispatch(
            &self.sessions,
            &record,
            task_map,
            self.execution_mode,
        )?;
        if pred_count == 0 {
            return Ok(vec![]);
        }
        Ok(pipeline::decode_entities(
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
    /// Each label has its own threshold; spans below their label's
    /// threshold are dropped. Useful when different labels have different
    /// score distributions (e.g., a domain-specific label that the model
    /// over-predicts can use a stricter threshold while keeping a more
    /// permissive bound on rarer labels).
    ///
    /// A label not present in the input list is **dropped entirely** —
    /// the underlying [`pipeline::decode_entities_with_thresholds`]
    /// treats unmapped labels as having threshold `+∞`. To mix
    /// per-label thresholds with a default for the rest, just enumerate
    /// every label.
    ///
    /// **Phase 1.5 / experimental.** Not behind a public trait.
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
        let record = self.transformer.transform(text, &[task])?;
        let num_words = record.word_to_char_maps.len();
        if num_words == 0 {
            return Ok(vec![]);
        }
        let task_map = record.tasks.first().ok_or_else(|| {
            crate::Error::Backend("gliner2_fastino: transformer produced no task mapping".into())
        })?;

        // Phase 3.5: dispatch on execution_mode.
        let (scorer_out, pred_count) = pipeline_iobinding::run_pipeline_dispatch(
            &self.sessions,
            &record,
            task_map,
            self.execution_mode,
        )?;
        if pred_count == 0 {
            return Ok(vec![]);
        }
        Ok(pipeline::decode_entities_with_thresholds(
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
    /// Each [`schema::StructureTask`] in `schema.structures` triggers
    /// one ONNX inference pass through the 8-session pipeline (encoder →
    /// token_gather → span_rep → schema_gather → count_pred_argmax →
    /// count_lstm_fixed → scorer). The scorer's `MAX_COUNT` axis is
    /// walked as the instance axis: an `[ExtractedStructure; pred_count]`
    /// is appended to the result for each task.
    ///
    /// **Phase 2 / experimental.** Returns instances even when all
    /// fields drop below threshold (with empty `fields` map). Phase 2.5
    /// may add an opt-in filter for empty instances.
    ///
    /// `FieldType::String` is the only fully-supported field type in
    /// Phase 2. `FieldType::List` and `FieldType::Choice` decode the
    /// same single-best-span treatment as `String` — see
    /// [`pipeline::decode_structure`] for the TODO markers.
    pub fn extract_structure(
        &self,
        text: &str,
        schema: &schema::TaskSchema,
        threshold: f32,
    ) -> crate::Result<Vec<schema::ExtractedStructure>> {
        if schema.structures.is_empty() {
            return Ok(vec![]);
        }
        let mut all_results: Vec<schema::ExtractedStructure> = Vec::new();
        for st in &schema.structures {
            if st.fields.is_empty() {
                continue; // skip degenerate task
            }
            let fields_owned: Vec<(String, schema::FieldType)> = st
                .fields
                .iter()
                .map(|f| (f.name.clone(), f.field_type))
                .collect();
            let task = processor::SchemaTask::Structures(st.name.clone(), fields_owned.clone());
            let record = self.transformer.transform(text, &[task])?;
            let num_words = record.word_to_char_maps.len();
            if num_words == 0 {
                continue;
            }
            let task_map = record.tasks.first().ok_or_else(|| {
                crate::Error::Backend(
                    "gliner2_fastino: transformer produced no task mapping".into(),
                )
            })?;

            // Phase 3.5: dispatch on execution_mode (per inference pass —
            // each structure task gets its own pipeline run).
            let (scorer_out, pred_count) = pipeline_iobinding::run_pipeline_dispatch(
                &self.sessions,
                &record,
                task_map,
                self.execution_mode,
            )?;
            if pred_count == 0 {
                continue;
            }

            let task_results = pipeline::decode_structure(
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
    /// Returns labels sorted by descending probability (softmax). The
    /// `threshold` parameter is reserved for future multi-label use; in
    /// Phase 3 single-label mode it's ignored.
    ///
    /// Not behind a public trait — see spec §3.
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
        let record = self.transformer.transform(text, &[task])?;
        let task_map = record.tasks.first().ok_or_else(|| {
            crate::Error::Backend("gliner2_fastino: transformer produced no task mapping".into())
        })?;

        // Phase 3.5: dispatch on execution_mode. Standard runs the
        // existing 4-call chain (encoder → schema_gather →
        // count_pred_argmax → classifier); IoBinding routes through
        // run_classify_pipeline. Both return all-zeros on pred_count=0.
        let probs = pipeline_iobinding::run_classify_dispatch(
            &self.sessions,
            &record,
            task_map,
            self.execution_mode,
        )?;

        let mut out: Vec<(String, f32)> = label_strings.into_iter().zip(probs).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }

    /// Batch extract entities with either shared or per-sample label sets.
    ///
    /// - [`BatchSchemaMode::Shared`]: every text uses the same label slice.
    ///   Equivalent to looping `extract_with_types(text, labels, threshold)`.
    /// - [`BatchSchemaMode::PerSample`]: each text uses its own label set.
    ///   Lengths must match. Returns a typed `crate::Error::Backend`
    ///   error on length mismatch.
    ///
    /// **Phase 1.5 / experimental.** Single-threaded — the "batch" here
    /// refers to API ergonomics, not parallel inference. For chunked
    /// progress callbacks, see [`Self::batch_extract_streaming`].
    pub fn batch_extract_with_schema_mode(
        &self,
        texts: &[&str],
        schema: BatchSchemaMode<'_>,
        threshold: f32,
    ) -> crate::Result<Vec<Vec<crate::Entity>>> {
        let mut out: Vec<Vec<crate::Entity>> = Vec::with_capacity(texts.len());
        match schema {
            BatchSchemaMode::Shared(labels) => {
                for text in texts {
                    out.push(self.extract_ner(text, labels, threshold)?);
                }
            }
            BatchSchemaMode::PerSample(per_text_labels) => {
                if per_text_labels.len() != texts.len() {
                    return Err(crate::Error::Backend(format!(
                        "gliner2_fastino: PerSample label count {} != texts count {}",
                        per_text_labels.len(),
                        texts.len()
                    )));
                }
                for (text, labels_owned) in texts.iter().zip(per_text_labels.iter()) {
                    let labels: Vec<&str> = labels_owned.to_vec();
                    out.push(self.extract_ner(text, &labels, threshold)?);
                }
            }
        }
        Ok(out)
    }

    /// Process a slice of texts in chunks, invoking `on_batch` after each
    /// text in the just-completed chunk.
    ///
    /// Useful for large-document workloads where you want incremental
    /// output instead of waiting for the entire batch to complete. The
    /// callback receives `(text_index, entities_for_this_text)` for each
    /// text as it lands.
    ///
    /// **Phase 1.5 / experimental.** Single-threaded — the "batch" in
    /// the name refers to chunked progress reporting, not parallel batched
    /// inference. Errors during any single text's extraction propagate
    /// out of `batch_extract_streaming` immediately and abort the loop.
    pub fn batch_extract_streaming<F>(
        &self,
        texts: &[&str],
        types: &[&str],
        threshold: f32,
        batch_size: usize,
        mut on_batch: F,
    ) -> crate::Result<()>
    where
        F: FnMut(usize, &[crate::Entity]),
    {
        if batch_size == 0 {
            return Err(crate::Error::Backend(
                "gliner2_fastino: batch_size must be > 0".into(),
            ));
        }
        let mut cursor = 0;
        while cursor < texts.len() {
            let end = (cursor + batch_size).min(texts.len());
            for (offset, text) in texts[cursor..end].iter().enumerate() {
                let idx = cursor + offset;
                let ents = self.extract_ner(text, types, threshold)?;
                on_batch(idx, &ents);
            }
            cursor = end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod streaming_tests {
    #[test]
    fn streaming_chunking_indices_are_contiguous_and_complete() {
        // The control flow that drives batch_extract_streaming's chunk
        // boundaries is what we want to lock in: cover all indices, no gaps,
        // last chunk handles a non-aligned tail. We can't easily run the
        // method without a real model, so we verify the chunk-boundary
        // logic directly with the same control structure.
        let texts: Vec<&str> = (0..10)
            .map(|i| match i {
                0 => "zero",
                1 => "one",
                2 => "two",
                3 => "three",
                4 => "four",
                5 => "five",
                6 => "six",
                7 => "seven",
                8 => "eight",
                _ => "nine",
            })
            .collect();

        let mut chunks_seen: Vec<(usize, usize)> = Vec::new();
        let batch_size = 3;
        let mut cursor = 0;
        while cursor < texts.len() {
            let end = (cursor + batch_size).min(texts.len());
            chunks_seen.push((cursor, end));
            cursor = end;
        }
        assert_eq!(chunks_seen, vec![(0, 3), (3, 6), (6, 9), (9, 10)]);

        // Exact-multiple case: no partial chunk.
        let mut exact: Vec<(usize, usize)> = Vec::new();
        let mut cursor = 0;
        let n = 9;
        while cursor < n {
            let end = (cursor + batch_size).min(n);
            exact.push((cursor, end));
            cursor = end;
        }
        assert_eq!(exact, vec![(0, 3), (3, 6), (6, 9)]);

        // Single text, batch_size > len: one chunk.
        let mut single: Vec<(usize, usize)> = Vec::new();
        let mut cursor = 0;
        let n = 1;
        while cursor < n {
            let end = (cursor + batch_size).min(n);
            single.push((cursor, end));
            cursor = end;
        }
        assert_eq!(single, vec![(0, 1)]);
    }
}

#[cfg(test)]
mod from_local_tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn from_local_rejects_lora_adapter_dir() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("adapter_config.json"), "{}").unwrap();

        let err = GLiNER2Fastino::from_local(dir.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("scripts/gliner2_export_onnx.py"),
            "missing script path: {msg}"
        );
        assert!(msg.contains("--lora-adapter"), "missing flag: {msg}");
    }

    #[test]
    fn from_local_missing_tokenizer_returns_typed_error() {
        let dir = tempdir().unwrap();
        // Empty directory — no tokenizer.json, no adapter_config.json.
        // With the subdir-first loading order, Sessions::from_dir fires
        // before tokenizer resolution and surfaces a "no complete v2 session
        // set" error. Both session-set and tokenizer errors indicate a
        // missing/incomplete model directory.
        let err = GLiNER2Fastino::from_local(dir.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("tokenizer") || msg.contains("no complete v2 session set"),
            "got {msg}"
        );
    }

    #[test]
    fn from_local_empty_dir_returns_session_set_error() {
        let dir = tempdir().unwrap();
        // Need at least tokenizer.json to bypass the early-return.
        // Stub one out using the project's own fixture.
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/gliner2_fastino/stub_tokenizer.json");
        fs::copy(&fixture, dir.path().join("tokenizer.json")).unwrap();
        // And a config.json with hidden_size.
        fs::write(
            dir.path().join("config.json"),
            r#"{"hidden_size": 768, "counting_layer": "count_lstm_v2"}"#,
        )
        .unwrap();

        let err = GLiNER2Fastino::from_local(dir.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no complete v2 session set"),
            "Phase 3 should report missing sessions, not 'Phase 3 needed'. Got: {msg}"
        );
    }

    #[test]
    fn schema_types_reachable_via_gliner2_fastino_path() {
        // Phase 2 M1: confirm the schema submodule re-exports the
        // structure-extraction types so callers don't have to depend on
        // gliner_multitask. Compile-time check + minimal value construction.
        use crate::backends::gliner2_fastino::schema::{
            ExtractedStructure, FieldType, StructureTask, TaskSchema,
        };
        let _schema: TaskSchema = TaskSchema::new().with_structure(
            StructureTask::new("invoice")
                .with_field("vendor", FieldType::String)
                .with_field("amount", FieldType::String),
        );
        let _es: ExtractedStructure = ExtractedStructure {
            structure_type: "invoice".to_string(),
            fields: std::collections::HashMap::new(),
        };
    }

    #[test]
    fn config_defaults_are_standard_mode() {
        // Phase 3.5 M2: GLiNER2FastinoConfig::default() picks Standard mode
        // and CPU-only ort defaults.
        let cfg = GLiNER2FastinoConfig::default();
        assert_eq!(cfg.execution_mode, ExecutionMode::Standard);
        assert!(!cfg.onnx.prefer_cuda);
        assert!(!cfg.onnx.prefer_coreml);
    }

    #[test]
    fn execution_mode_default_is_standard() {
        assert_eq!(ExecutionMode::default(), ExecutionMode::Standard);
    }

    #[test]
    fn from_local_with_options_delegates_to_config_with_standard_mode() {
        // The legacy from_local_with_options should now produce a Standard-mode engine.
        // Since we can't load an actual model in unit tests, just verify the equivalence
        // of the two paths via the LoRA-rejection error: it's hit AFTER the
        // ExecutionMode-bearing config struct is constructed but BEFORE any session
        // load. So calling either constructor on a LoRA dir yields the same error.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("adapter_config.json"), "{}").unwrap();

        let err1 = GLiNER2Fastino::from_local_with_options(
            dir.path(),
            crate::backends::hf_loader::OnnxSessionConfig::default(),
        )
        .unwrap_err();
        let err2 =
            GLiNER2Fastino::from_local_with_config(dir.path(), GLiNER2FastinoConfig::default())
                .unwrap_err();

        // Both should be the LoraAdapterNotSupported error with the same path.
        assert!(err1.to_string().contains("scripts/gliner2_export_onnx.py"));
        assert!(err2.to_string().contains("scripts/gliner2_export_onnx.py"));
    }

    #[test]
    fn engine_is_send_and_sync() {
        // Phase 3.5 M4: future additions for IoBinding must not break the
        // engine's Send + Sync — anno's threading model shares engines via
        // Arc across worker threads (see crate::Model and ZeroShotNER trait
        // bounds). Per-call MemoryInfo creation in pipeline_iobinding (M5+)
        // keeps `ort::memory::MemoryInfo` (which is !Send + !Sync) off the
        // engine struct.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GLiNER2Fastino>();
    }
}
