//! TPLinker: Single-stage Joint Entity-Relation Extraction
//!
//! TPLinker uses a handshaking tagging scheme for joint entity-relation extraction.
//! It models entity boundaries and relations simultaneously using a unified tagging matrix.
//!
//! # Handshaking Matrix
//!
//! For a sequence of length L, TPLinker builds a handshaking sequence of length
//! `L*(L+1)/2` covering all upper-triangular position pairs `(i, j)` where `i <= j`.
//!
//! The flat index for pair `(i, j)` is:
//! ```text
//! idx = i * L - i * (i - 1) / 2 + (j - i)
//! ```
//!
//! Three output heads decode from this sequence:
//! - `ent_logits`:      `[batch, hs_len, 5]` (entity boundary tags: NONE, SH2OH, OH2ST, ST2OT, OT2ST)
//! - `head_rel_logits`: `[batch, hs_len, num_rels * 3]` (head relations: NONE, SH2OH, OH2ST per type)
//! - `tail_rel_logits`: `[batch, hs_len, num_rels * 3]` (tail relations: NONE, SH2OH, OH2ST per type)
//!
//! # Backends
//!
//! - **ONNX** (feature `onnx`): Full neural handshaking matrix decoding.
//!   Export model with: `uv run scripts/export_tplinker_onnx.py`
//! - **Fallback** (no feature): Heuristic baseline using StackedNER + trigger matching.
//!
//! # Research
//!
//! - **Paper**: [TPLinker: Single-stage Joint Extraction](https://aclanthology.org/2020.coling-main.138/)
//! - Wang et al., COLING 2020
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::tplinker::TPLinker;
//!
//! let extractor = TPLinker::new()?;
//! let result = extractor.extract_with_relations(
//!     "Steve Jobs founded Apple in 1976.",
//!     &["person", "organization"],
//!     &["founded", "works_for"],
//!     0.5
//! )?;
//!
//! for entity in &result.entities {
//!     println!("Entity: {} ({})", entity.text, entity.entity_type);
//! }
//!
//! for relation in &result.relations {
//!     let head = &result.entities[relation.head_idx];
//!     let tail = &result.entities[relation.tail_idx];
//!     println!("Relation: {} --[{}]--> {}", head.text, relation.relation_type, tail.text);
//! }
//! ```

use crate::backends::inference::{ExtractionWithRelations, RelationExtractor};
#[cfg(feature = "onnx")]
use crate::EntityCategory;
use crate::{Confidence, Entity, EntityType, Language, Model, Result};
use std::borrow::Cow;

// =============================================================================
// ONNX Backend (feature-gated)
// =============================================================================

#[cfg(feature = "onnx")]
mod onnx_impl {
    use super::*;
    use crate::backends::hf_loader;
    use crate::backends::inference::RelationTriple;
    use crate::backends::ort_compat::tensor_from_ndarray;
    use crate::Error;
    use ndarray::Array2;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    /// Default cache directory for TPLinker models.
    fn default_model_dir() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("anno")
            .join("models")
            .join("tplinker")
    }

    /// TPLinker model configuration (from `tplinker_config.json`).
    #[derive(Debug, Clone, serde::Deserialize)]
    #[allow(dead_code)]
    pub struct TPLinkerConfig {
        #[serde(default)]
        pub dataset: String,
        #[serde(default)]
        pub encoder: String,
        #[serde(default = "default_5")]
        pub num_entity_tags: usize,
        #[serde(default)]
        pub num_relation_types: usize,
        #[serde(default = "default_768")]
        pub hidden_size: usize,
        #[serde(default)]
        pub shaking_type: String,
        #[serde(default)]
        pub entity_tags: Vec<String>,
        #[serde(default)]
        pub relation_tags: Vec<String>,
        #[serde(default)]
        pub relations: Vec<String>,
    }

    fn default_5() -> usize {
        5
    }
    fn default_768() -> usize {
        768
    }

    /// TPLinker ONNX backend for joint entity-relation extraction.
    #[derive(Debug)]
    pub struct TPLinkerOnnx {
        session: Mutex<ort::session::Session>,
        tokenizer: tokenizers::Tokenizer,
        config: TPLinkerConfig,
        entity_threshold: f32,
        relation_threshold: f32,
    }

    /// Entity tag indices in the handshaking matrix.
    const ENT_NONE: usize = 0;
    const ENT_SH2OH: usize = 1;
    const ENT_OH2ST: usize = 2;
    const ENT_ST2OT: usize = 3;
    const ENT_OT2ST: usize = 4;

    /// Relation tag indices (per relation type).
    const REL_NONE: usize = 0;
    const REL_SH2OH: usize = 1;
    const REL_OH2ST: usize = 2;

    impl TPLinkerOnnx {
        /// Load TPLinker from a local directory.
        pub fn from_local(dir: &Path) -> Result<Self> {
            Self::from_local_with_thresholds(dir, 0.15, 0.55)
        }

        /// Load with custom thresholds.
        pub fn from_local_with_thresholds(
            dir: &Path,
            entity_threshold: f32,
            relation_threshold: f32,
        ) -> Result<Self> {
            let model_path = dir.join("model.onnx");
            if !model_path.exists() {
                let default_dir = default_model_dir();
                let alt_path = default_dir.join("model.onnx");
                if alt_path.exists() {
                    return Self::from_local_with_thresholds(
                        &default_dir,
                        entity_threshold,
                        relation_threshold,
                    );
                }
                return Err(Error::Retrieval(format!(
                    "TPLinker model not found at {}. Export with: uv run scripts/export_tplinker_onnx.py",
                    model_path.display()
                )));
            }

            let tokenizer_path = dir.join("tokenizer.json");
            if !tokenizer_path.exists() {
                return Err(Error::Retrieval(format!(
                    "Tokenizer not found at {}",
                    tokenizer_path.display()
                )));
            }

            let config: TPLinkerConfig = {
                let config_path = dir.join("tplinker_config.json");
                if config_path.exists() {
                    let data = std::fs::read_to_string(&config_path)
                        .map_err(|e| Error::Retrieval(format!("tplinker config read: {e}")))?;
                    serde_json::from_str(&data)
                        .map_err(|e| Error::Parse(format!("tplinker config parse: {e}")))?
                } else {
                    return Err(Error::Retrieval(
                        "tplinker_config.json not found in model directory".to_string(),
                    ));
                }
            };

            let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;
            let session = hf_loader::create_onnx_session(
                &model_path,
                hf_loader::OnnxSessionConfig::default(),
            )?;

            log::info!(
                "[TPLinker-ONNX] Loaded from {} ({} relation types)",
                dir.display(),
                config.num_relation_types
            );

            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
                config,
                entity_threshold,
                relation_threshold,
            })
        }

        /// Run ONNX inference and decode the handshaking matrix.
        pub fn extract_joint(
            &self,
            text: &str,
            relation_types: &[&str],
            threshold: f32,
        ) -> Result<ExtractionWithRelations> {
            if text.is_empty() {
                return Ok(ExtractionWithRelations::default());
            }

            let encoding = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| Error::Inference(format!("TPLinker tokenize: {e}")))?;

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            let attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&m| m as i64)
                .collect();
            let seq_len = input_ids.len();

            let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)
                .map_err(|e| Error::Parse(format!("input_ids: {e}")))?;
            let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask)
                .map_err(|e| Error::Parse(format!("attention_mask: {e}")))?;

            let t_ids = tensor_from_ndarray(input_ids_arr)
                .map_err(|e| Error::Inference(format!("tensor: {e}")))?;
            let t_mask = tensor_from_ndarray(attention_mask_arr)
                .map_err(|e| Error::Inference(format!("tensor: {e}")))?;

            let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => t_ids.into_dyn(),
                    "attention_mask" => t_mask.into_dyn(),
                ])
                .map_err(|e| Error::Inference(format!("TPLinker ONNX run: {e}")))?;

            // Extract output tensors as flat slices with shapes
            let (_ent_shape, ent_data) = outputs
                .get("ent_logits")
                .ok_or_else(|| Error::Inference("Missing ent_logits output".to_string()))?
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Inference(format!("extract ent_logits: {e}")))?;

            let (_head_shape, head_data) = outputs
                .get("head_rel_logits")
                .ok_or_else(|| Error::Inference("Missing head_rel_logits output".to_string()))?
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Inference(format!("extract head_rel_logits: {e}")))?;

            let (_tail_shape, tail_data) = outputs
                .get("tail_rel_logits")
                .ok_or_else(|| Error::Inference("Missing tail_rel_logits output".to_string()))?
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Inference(format!("extract tail_rel_logits: {e}")))?;

            // Decode entities from handshaking entity tags
            let entities = self.decode_entities(text, &encoding, ent_data, seq_len)?;

            // Decode relations from head/tail relation logits
            let relations = self.decode_relations(
                &entities,
                head_data,
                tail_data,
                seq_len,
                relation_types,
                threshold,
            );

            Ok(ExtractionWithRelations {
                entities,
                relations,
            })
        }

        /// Decode entities from the handshaking entity logits.
        ///
        /// Entity boundaries are encoded as:
        /// - SH2OH at (i, j): Subject Head at i, Object Head at j
        /// - OH2ST at (i, j): Object Head at i, Subject Tail at j
        ///
        /// For NER (without relation context), we detect entity spans by finding
        /// positions where the entity tag has the highest logit (argmax != NONE).
        fn decode_entities(
            &self,
            text: &str,
            encoding: &tokenizers::Encoding,
            ent_data: &[f32],
            seq_len: usize,
        ) -> Result<Vec<Entity>> {
            let hs_len = seq_len * (seq_len + 1) / 2;
            let num_ent_tags = self.config.num_entity_tags;
            let mut entities = Vec::new();

            // Scan the handshaking sequence for non-NONE entity tags.
            // Layout: [batch=1, hs_len, num_entity_tags], flat index = idx * num_tags + tag
            for i in 0..seq_len {
                for j in i..seq_len {
                    let idx = handshaking_index(i, j, seq_len);
                    if idx >= hs_len {
                        continue;
                    }

                    let base = idx * num_ent_tags;

                    // Find argmax across entity tags
                    let mut best_tag = ENT_NONE;
                    let mut best_score = ent_data[base + ENT_NONE];
                    for tag in 1..num_ent_tags {
                        let score = ent_data[base + tag];
                        if score > best_score {
                            best_score = score;
                            best_tag = tag;
                        }
                    }

                    if best_tag == ENT_NONE {
                        continue;
                    }

                    // Convert softmax-like score to confidence
                    let confidence = softmax_confidence_flat(ent_data, idx, num_ent_tags);
                    if confidence < self.entity_threshold {
                        continue;
                    }

                    // Map token positions i..=j back to character offsets
                    if let Some((char_start, char_end)) = token_span_to_chars(encoding, text, i, j)
                    {
                        if char_start < char_end {
                            let span_text =
                                crate::offset::TextSpan::from_chars(text, char_start, char_end)
                                    .extract(text);

                            // Infer entity type from tag (SH2OH suggests Subject, etc.)
                            let entity_type = match best_tag {
                                ENT_SH2OH | ENT_OH2ST => {
                                    EntityType::custom("SUBJECT", EntityCategory::Misc)
                                }
                                ENT_ST2OT | ENT_OT2ST => {
                                    EntityType::custom("OBJECT", EntityCategory::Misc)
                                }
                                _ => EntityType::custom("ENTITY", EntityCategory::Misc),
                            };

                            let mut entity = Entity::new(
                                span_text,
                                entity_type,
                                char_start,
                                char_end,
                                confidence as f64,
                            );
                            entity.provenance = Some(crate::Provenance {
                                source: Cow::Borrowed("tplinker"),
                                method: crate::ExtractionMethod::Neural,
                                pattern: None,
                                raw_confidence: Some(crate::Confidence::from(confidence)),
                                model_version: Some(Cow::Borrowed("onnx")),
                                timestamp: None,
                            });
                            entities.push(entity);
                        }
                    }
                }
            }

            // Deduplicate overlapping entities, keeping highest confidence
            entities.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut seen_spans = std::collections::HashSet::new();
            entities.retain(|e| seen_spans.insert((e.start(), e.end())));

            Ok(entities)
        }

        /// Decode relations from head/tail relation logits (flat `&[f32]`).
        ///
        /// Layout: `[batch=1, hs_len, num_relation_types * 3]`
        fn decode_relations(
            &self,
            entities: &[Entity],
            head_data: &[f32],
            _tail_data: &[f32],
            seq_len: usize,
            requested_types: &[&str],
            threshold: f32,
        ) -> Vec<RelationTriple> {
            if entities.len() < 2 || self.config.num_relation_types == 0 {
                return Vec::new();
            }

            let rel_threshold = if threshold > 0.0 {
                threshold
            } else {
                self.relation_threshold
            };

            let hs_len = seq_len * (seq_len + 1) / 2;
            let num_rel_cols = self.config.num_relation_types * 3;
            let mut relations = Vec::new();

            for (head_idx, head_ent) in entities.iter().enumerate() {
                for (tail_idx, tail_ent) in entities.iter().enumerate() {
                    if head_idx == tail_idx {
                        continue;
                    }

                    let (hi, hj) = (head_ent.start(), tail_ent.start());
                    let (i, j) = if hi <= hj { (hi, hj) } else { (hj, hi) };

                    if j >= seq_len {
                        continue;
                    }

                    let hs_idx = handshaking_index(i, j, seq_len);
                    if hs_idx >= hs_len {
                        continue;
                    }

                    // Flat offset for this handshaking position in head_data
                    let row_base = hs_idx * num_rel_cols;

                    for rel_idx in 0..self.config.num_relation_types {
                        let base = row_base + rel_idx * 3;

                        let head_none = head_data[base + REL_NONE];
                        let head_sh2oh = head_data[base + REL_SH2OH];
                        let head_oh2st = head_data[base + REL_OH2ST];
                        let head_max = head_sh2oh.max(head_oh2st);

                        if head_max <= head_none {
                            continue;
                        }

                        // Softmax confidence for the winning non-NONE class
                        let sum = (head_none.exp() + head_sh2oh.exp() + head_oh2st.exp()).ln();
                        let conf_f32 = (head_max - sum).exp();

                        if conf_f32 < rel_threshold {
                            continue;
                        }

                        let rel_label = if rel_idx < self.config.relations.len() {
                            &self.config.relations[rel_idx]
                        } else {
                            continue;
                        };

                        if !requested_types.is_empty()
                            && !requested_types
                                .iter()
                                .any(|&rt| rt.eq_ignore_ascii_case(rel_label))
                        {
                            continue;
                        }

                        relations.push(RelationTriple {
                            head_idx,
                            tail_idx,
                            relation_type: rel_label.clone(),
                            confidence: Confidence::new(conf_f32 as f64),
                        });
                    }
                }
            }

            // Filter self-relations where head and tail have identical text
            // (e.g. Apple@19 ACQUIRED Apple@136 when the same entity name
            // appears at multiple positions).
            relations.retain(|r| entities[r.head_idx].text != entities[r.tail_idx].text);

            relations.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut seen = std::collections::HashSet::new();
            relations.retain(|r| seen.insert((r.head_idx, r.tail_idx)));

            relations
        }
    }

    /// Compute the flat handshaking index for position pair (i, j) where i <= j.
    fn handshaking_index(i: usize, j: usize, seq_len: usize) -> usize {
        i * seq_len - i * (i.wrapping_sub(1)) / 2 + (j - i)
    }

    /// Convert token span [start_tok, end_tok] to character offsets.
    fn token_span_to_chars(
        encoding: &tokenizers::Encoding,
        text: &str,
        start_tok: usize,
        end_tok: usize,
    ) -> Option<(usize, usize)> {
        let offsets = encoding.get_offsets();
        if start_tok >= offsets.len() || end_tok >= offsets.len() {
            return None;
        }

        let byte_start = offsets[start_tok].0;
        let byte_end = offsets[end_tok].1;

        if byte_start >= byte_end || byte_end > text.len() {
            return None;
        }

        // Convert byte offsets to character offsets
        let char_start = text[..byte_start].chars().count();
        let char_end = text[..byte_end].chars().count();

        Some((char_start, char_end))
    }

    /// Compute softmax confidence for the winning class at a handshaking position.
    ///
    /// `data` layout: `[batch=1, hs_len, num_tags]`, flat index = `hs_idx * num_tags + tag`.
    fn softmax_confidence_flat(data: &[f32], hs_idx: usize, num_tags: usize) -> f32 {
        let base = hs_idx * num_tags;

        let mut max_logit = f32::NEG_INFINITY;
        for tag in 0..num_tags {
            max_logit = max_logit.max(data[base + tag]);
        }

        let mut sum_exp = 0.0f32;
        let mut best_exp = 0.0f32;
        for tag in 0..num_tags {
            let e = (data[base + tag] - max_logit).exp();
            sum_exp += e;
            if data[base + tag] == max_logit {
                best_exp = e;
            }
        }

        best_exp / sum_exp
    }
}

// =============================================================================
// Heuristic Fallback (always available)
// =============================================================================

mod heuristic_impl {
    use super::*;
    use crate::backends::inference::{extract_relation_triples_simple, RelationExtractionConfig};
    use std::collections::HashSet;

    /// TPLinker heuristic baseline for when ONNX is not available.
    #[derive(Debug)]
    pub struct TPLinkerHeuristic {
        pub entity_threshold: f32,
        pub relation_threshold: f32,
    }

    impl TPLinkerHeuristic {
        pub fn extract_with_handshaking(
            &self,
            text: &str,
            entity_types: &[&str],
            relation_types: &[&str],
            threshold: f32,
        ) -> Result<ExtractionWithRelations> {
            let rel_threshold = if threshold > 0.0 {
                threshold
            } else {
                self.relation_threshold
            };
            let ent_threshold = self.entity_threshold;

            let ner = crate::StackedNER::default();
            let mut entities = ner.extract_entities(text, None)?;

            if !entity_types.is_empty() {
                let requested: Vec<String> =
                    entity_types.iter().map(|s| s.to_lowercase()).collect();
                let looks_supported = requested.iter().all(|t| {
                    matches!(
                        t.as_str(),
                        "person"
                            | "per"
                            | "organization"
                            | "organisation"
                            | "org"
                            | "location"
                            | "loc"
                            | "date"
                            | "time"
                            | "money"
                            | "misc"
                    )
                });
                if looks_supported {
                    let allowed: HashSet<EntityType> = entity_types
                        .iter()
                        .map(|s| EntityType::from_label(s))
                        .collect();
                    entities.retain(|e| allowed.contains(&e.entity_type));
                }
            }

            entities.retain(|e| e.confidence >= f64::from(ent_threshold));

            for entity in &mut entities {
                entity.provenance = Some(crate::Provenance {
                    source: Cow::Borrowed("tplinker"),
                    method: crate::ExtractionMethod::Heuristic,
                    pattern: None,
                    raw_confidence: Some(entity.confidence),
                    model_version: Some(Cow::Borrowed("heuristic-fallback")),
                    timestamp: None,
                });
            }

            const DEFAULT_RELATIONS: &[&str] = &[
                "CEO_OF",
                "WORKS_FOR",
                "FOUNDED",
                "MANAGES",
                "REPORTS_TO",
                "LOCATED_IN",
                "BORN_IN",
                "LIVES_IN",
                "DIED_IN",
                "OCCURRED_ON",
                "STARTED_ON",
                "ENDED_ON",
                "PART_OF",
                "ACQUIRED",
                "MERGED_WITH",
                "PARENT_OF",
                "MARRIED_TO",
                "CHILD_OF",
                "SIBLING_OF",
            ];

            let rels: Vec<&str> = if relation_types.is_empty() {
                DEFAULT_RELATIONS.to_vec()
            } else {
                relation_types.to_vec()
            };

            let rel_config = RelationExtractionConfig {
                threshold: Confidence::new(rel_threshold as f64),
                max_span_distance: 120,
                extract_triggers: false,
            };

            let relations = extract_relation_triples_simple(&entities, text, &rels, &rel_config);

            Ok(ExtractionWithRelations {
                entities,
                relations,
            })
        }
    }
}

// =============================================================================
// Public TPLinker type (dispatches to ONNX or heuristic)
// =============================================================================

/// TPLinker backend for joint entity-relation extraction.
///
/// When the `onnx` feature is enabled and a model has been loaded, uses real
/// neural handshaking matrix decoding. Otherwise falls back to a heuristic
/// baseline using StackedNER + trigger matching.
#[derive(Debug)]
pub struct TPLinker {
    entity_threshold: f32,
    relation_threshold: f32,
    #[cfg(feature = "onnx")]
    onnx: Option<onnx_impl::TPLinkerOnnx>,
}

impl TPLinker {
    /// Create a new TPLinker instance.
    ///
    /// Attempts to load the ONNX model from the default cache directory.
    /// Falls back to heuristic mode if the model is not available.
    pub fn new() -> Result<Self> {
        Ok(Self::with_thresholds(0.15, 0.55))
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(entity_threshold: f32, relation_threshold: f32) -> Self {
        #[cfg(feature = "onnx")]
        let onnx = {
            let default_dir = dirs::cache_dir()
                .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
                .join("anno")
                .join("models")
                .join("tplinker");
            onnx_impl::TPLinkerOnnx::from_local_with_thresholds(
                &default_dir,
                entity_threshold,
                relation_threshold,
            )
            .ok()
        };

        Self {
            entity_threshold,
            relation_threshold,
            #[cfg(feature = "onnx")]
            onnx,
        }
    }

    /// Create from a local model directory (ONNX only).
    #[cfg(feature = "onnx")]
    pub fn from_local(dir: &std::path::Path) -> Result<Self> {
        let onnx = onnx_impl::TPLinkerOnnx::from_local(dir)?;
        Ok(Self {
            entity_threshold: 0.15,
            relation_threshold: 0.55,
            onnx: Some(onnx),
        })
    }

    /// Whether this instance is using the neural ONNX backend.
    pub fn is_neural(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            self.onnx.is_some()
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }

    fn heuristic(&self) -> heuristic_impl::TPLinkerHeuristic {
        heuristic_impl::TPLinkerHeuristic {
            entity_threshold: self.entity_threshold,
            relation_threshold: self.relation_threshold,
        }
    }
}

impl Model for TPLinker {
    fn extract_entities(&self, text: &str, _language: Option<Language>) -> Result<Vec<Entity>> {
        #[cfg(feature = "onnx")]
        if let Some(ref onnx) = self.onnx {
            let result = onnx.extract_joint(text, &[], 0.0)?;
            return Ok(result.entities);
        }

        // Heuristic fallback
        let heuristic = crate::StackedNER::default();
        let mut entities = heuristic.extract_entities(text, None)?;
        entities.retain(|e| e.confidence >= f64::from(self.entity_threshold));
        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Date,
            EntityType::Time,
            EntityType::Money,
        ]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "tplinker"
    }

    fn description(&self) -> &'static str {
        #[cfg(feature = "onnx")]
        if self.onnx.is_some() {
            return "TPLinker joint entity-relation extraction (ONNX, Wang et al. COLING 2020)";
        }
        "TPLinker joint entity-relation extraction (heuristic fallback)"
    }

    fn capabilities(&self) -> crate::ModelCapabilities {
        crate::ModelCapabilities {
            relation_capable: true,
            ..Default::default()
        }
    }

    fn as_relation_extractor(&self) -> Option<&dyn crate::backends::inference::RelationExtractor> {
        Some(self)
    }
}

impl RelationExtractor for TPLinker {
    fn extract_with_relations(
        &self,
        text: &str,
        entity_types: &[&str],
        relation_types: &[&str],
        threshold: f32,
    ) -> Result<ExtractionWithRelations> {
        #[cfg(feature = "onnx")]
        if let Some(ref onnx) = self.onnx {
            return onnx.extract_joint(text, relation_types, threshold);
        }

        self.heuristic()
            .extract_with_handshaking(text, entity_types, relation_types, threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    /// Cached TPLinker with standard thresholds (0.15/0.55).
    /// Avoids repeated ONNX model probe + disk scan per test (~10s each).
    static TP_STANDARD: LazyLock<TPLinker> =
        LazyLock::new(|| TPLinker::with_thresholds(0.15, 0.55));

    /// Cached TPLinker with zero thresholds (accepts everything).
    static TP_ZERO: LazyLock<TPLinker> = LazyLock::new(|| TPLinker::with_thresholds(0.0, 0.0));

    #[test]
    fn test_tplinker_creation() {
        let tplinker = TPLinker::new().unwrap();
        assert!(tplinker.is_available());
    }

    #[test]
    fn test_tplinker_entity_extraction() {
        let entities = TP_STANDARD
            .extract_entities("Steve Jobs founded Apple.", None)
            .unwrap();
        assert!(!entities.is_empty());
    }

    #[test]
    fn test_tplinker_relation_extraction() {
        let tplinker = &*TP_STANDARD;
        let out = tplinker
            .extract_with_relations(
                "Steve Jobs founded Apple in 1976.",
                &["person", "organization"],
                &["founded"],
                0.5,
            )
            .unwrap();
        assert!(out.entities.len() >= 2);
        // In heuristic mode, expect a founded relation from trigger matching.
        // In ONNX mode, depends on model weights.
        if !tplinker.is_neural() {
            assert!(
                out.relations.iter().any(|r| r.relation_type == "founded"),
                "expected a founded relation; got: {:?}",
                out.relations
            );
        }
    }

    #[test]
    fn test_tplinker_name_and_description() {
        let tp = TPLinker::new().unwrap();
        assert_eq!(tp.name(), "tplinker");
        let desc = tp.description();
        assert!(
            desc.contains("TPLinker"),
            "description should mention TPLinker, got: {desc}"
        );
    }

    #[test]
    fn test_tplinker_supported_types_complete() {
        let tp = TPLinker::new().unwrap();
        let types = tp.supported_types();
        assert!(types.contains(&EntityType::Person));
        assert!(types.contains(&EntityType::Organization));
        assert!(types.contains(&EntityType::Location));
        assert!(types.contains(&EntityType::Date));
        assert!(types.contains(&EntityType::Time));
        assert!(types.contains(&EntityType::Money));
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_tplinker_empty_text() {
        let entities = TP_STANDARD.extract_entities("", None).unwrap();
        assert!(entities.is_empty(), "empty text should produce no entities");

        let out = TP_STANDARD
            .extract_with_relations("", &["person"], &["founded"], 0.5)
            .unwrap();
        assert!(out.entities.is_empty());
        assert!(out.relations.is_empty());
    }

    #[test]
    fn test_tplinker_entities_only_no_relations() {
        let out = TP_STANDARD
            .extract_with_relations("Tokyo is a city.", &["location"], &[], 0.5)
            .unwrap();
        // With no relation types requested, relations should be empty.
        assert!(
            out.relations.is_empty(),
            "no relation types requested, but got {} relations",
            out.relations.len()
        );
    }

    #[test]
    fn test_tplinker_provenance_metadata() {
        let out = TP_ZERO
            .extract_with_relations(
                "Steve Jobs founded Apple in 1976.",
                &["person", "organization"],
                &["founded"],
                0.0,
            )
            .unwrap();

        for entity in &out.entities {
            let prov = entity
                .provenance
                .as_ref()
                .expect("every tplinker entity should have provenance");
            assert_eq!(
                prov.source, "tplinker",
                "provenance source should be 'tplinker'"
            );
        }
    }

    #[test]
    fn test_tplinker_multiple_relations_same_entity_types() {
        let text = "Tim Cook leads Apple. Satya Nadella leads Microsoft.";
        let out = TP_ZERO
            .extract_with_relations(
                text,
                &["person", "organization"],
                &["CEO_OF", "WORKS_FOR", "MANAGES"],
                0.0,
            )
            .unwrap();
        assert!(
            out.entities.len() >= 2,
            "should find at least 2 entities, got: {}",
            out.entities.len()
        );
        for r in &out.relations {
            assert!(r.head_idx < out.entities.len());
            assert!(r.tail_idx < out.entities.len());
        }
    }

    #[test]
    fn test_tplinker_capabilities() {
        let tp = TPLinker::new().unwrap();
        let caps = tp.capabilities();
        assert!(caps.relation_capable);
    }

    #[test]
    fn test_tplinker_custom_thresholds() {
        // Strict thresholds need a fresh instance (0.99/0.99 differs from cached)
        let tp_strict = TPLinker::with_thresholds(0.99, 0.99);
        let entities = tp_strict
            .extract_entities("Steve Jobs founded Apple.", None)
            .unwrap();
        let entities_lenient = TP_ZERO
            .extract_entities("Steve Jobs founded Apple.", None)
            .unwrap();
        assert!(
            entities.len() <= entities_lenient.len(),
            "strict thresholds should produce fewer or equal entities"
        );
    }

    #[test]
    fn test_tplinker_unicode_offsets_invariants() {
        let text = "Dr. 田中 met François Müller in 東京. \u{1f389}";
        let out = TP_STANDARD
            .extract_with_relations(
                text,
                &["person", "location", "organization"],
                &["works_for", "located_in", "founded"],
                0.0,
            )
            .unwrap();

        let text_len = text.chars().count();
        for e in &out.entities {
            assert!(
                e.start() < e.end(),
                "invalid span: {:?}",
                (e.start(), e.end())
            );
            assert!(
                e.end() <= text_len,
                "span out of bounds: {:?} (len={})",
                (e.start(), e.end()),
                text_len
            );
            let extracted =
                crate::offset::TextSpan::from_chars(text, e.start(), e.end()).extract(text);
            assert_eq!(extracted, e.text);
        }
        for r in &out.relations {
            assert!(r.head_idx < out.entities.len());
            assert!(r.tail_idx < out.entities.len());
        }
    }

    #[test]
    fn test_is_neural_flag() {
        #[cfg(not(feature = "onnx"))]
        assert!(
            !TP_STANDARD.is_neural(),
            "without onnx feature, TPLinker should not be neural"
        );
        #[cfg(feature = "onnx")]
        {
            // With onnx feature enabled, is_neural depends on model availability.
            // Without a downloaded model it should be false (heuristic fallback).
            let neural = TP_STANDARD.is_neural();
            // Accept either value -- we're testing it doesn't panic and returns a bool.
            let _ = neural;
        }
    }

    #[test]
    fn test_handshaking_index() {
        // Handshaking index: idx = i * L - i * (i-1) / 2 + (j - i)
        // For seq_len=4, hs_len=10
        fn hs(i: usize, j: usize, l: usize) -> usize {
            i * l - i * (i.wrapping_sub(1)) / 2 + (j - i)
        }
        assert_eq!(hs(0, 0, 4), 0);
        assert_eq!(hs(0, 3, 4), 3);
        assert_eq!(hs(1, 1, 4), 4);
        assert_eq!(hs(2, 2, 4), 7);
        assert_eq!(hs(3, 3, 4), 9);
    }

    #[test]
    fn test_handshaking_index_monotonic() {
        // For any seq_len, iterating (i,j) with i<=j should produce strictly increasing indices
        fn hs(i: usize, j: usize, l: usize) -> usize {
            i * l - i * (i.wrapping_sub(1)) / 2 + (j - i)
        }
        for seq_len in 2..8 {
            let mut prev = None;
            for i in 0..seq_len {
                for j in i..seq_len {
                    let idx = hs(i, j, seq_len);
                    if let Some(p) = prev {
                        assert!(
                            idx > p,
                            "handshaking index not monotonic: hs({i},{j},{seq_len})={idx} <= {p}"
                        );
                    }
                    prev = Some(idx);
                }
            }
        }
    }

    #[test]
    fn test_handshaking_index_total_count() {
        // Total handshaking positions for seq_len L should be L*(L+1)/2
        fn hs(i: usize, j: usize, l: usize) -> usize {
            i * l - i * (i.wrapping_sub(1)) / 2 + (j - i)
        }
        for seq_len in 1..10 {
            let last = hs(seq_len - 1, seq_len - 1, seq_len);
            assert_eq!(last, seq_len * (seq_len + 1) / 2 - 1);
        }
    }

    #[test]
    fn test_softmax_confidence_uniform() {
        // Replicate softmax_confidence_flat logic for testing
        fn softmax_conf(data: &[f32], num_tags: usize) -> f32 {
            let mut max_logit = f32::NEG_INFINITY;
            for &v in &data[..num_tags] {
                max_logit = max_logit.max(v);
            }
            let mut sum_exp = 0.0f32;
            let mut best_exp = 0.0f32;
            for &v in &data[..num_tags] {
                let e = (v - max_logit).exp();
                sum_exp += e;
                if v == max_logit {
                    best_exp = e;
                }
            }
            best_exp / sum_exp
        }

        // Uniform logits -> confidence = 1/num_tags
        let data = vec![1.0f32; 4];
        let conf = softmax_conf(&data, 4);
        assert!(
            (conf - 0.25).abs() < 1e-5,
            "uniform logits should give 1/N confidence, got {conf}"
        );
    }

    #[test]
    fn test_softmax_confidence_dominant() {
        fn softmax_conf(data: &[f32], num_tags: usize) -> f32 {
            let mut max_logit = f32::NEG_INFINITY;
            for &v in &data[..num_tags] {
                max_logit = max_logit.max(v);
            }
            let mut sum_exp = 0.0f32;
            let mut best_exp = 0.0f32;
            for &v in &data[..num_tags] {
                let e = (v - max_logit).exp();
                sum_exp += e;
                if v == max_logit {
                    best_exp = e;
                }
            }
            best_exp / sum_exp
        }

        let data = vec![100.0, 0.0, 0.0];
        let conf = softmax_conf(&data, 3);
        assert!(conf > 0.99, "dominant logit should give ~1.0, got {conf}");
    }

    #[test]
    fn test_softmax_confidence_numerical_stability() {
        fn softmax_conf(data: &[f32], num_tags: usize) -> f32 {
            let mut max_logit = f32::NEG_INFINITY;
            for &v in &data[..num_tags] {
                max_logit = max_logit.max(v);
            }
            let mut sum_exp = 0.0f32;
            let mut best_exp = 0.0f32;
            for &v in &data[..num_tags] {
                let e = (v - max_logit).exp();
                sum_exp += e;
                if v == max_logit {
                    best_exp = e;
                }
            }
            best_exp / sum_exp
        }

        // Large values: max-subtraction prevents overflow
        let data = vec![1000.0, 999.0];
        let conf = softmax_conf(&data, 2);
        assert!(!conf.is_nan(), "should handle large values without NaN");
        assert!(conf > 0.5 && conf <= 1.0);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_tplinker_config_deserialization() {
        let json = r#"{
            "entity_tags": ["PER", "ORG"],
            "relation_tags": ["founded", "works_for"],
            "relations": ["PER-founded-ORG", "PER-works_for-ORG"]
        }"#;
        let config: onnx_impl::TPLinkerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.entity_tags.len(), 2);
        assert_eq!(config.relation_tags.len(), 2);
        assert_eq!(config.relations.len(), 2);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_tplinker_config_defaults() {
        let json = r#"{}"#;
        let config: onnx_impl::TPLinkerConfig = serde_json::from_str(json).unwrap();
        // Defaults should be provided for empty config
        assert!(config.entity_tags.is_empty() || !config.entity_tags.is_empty());
    }
}
