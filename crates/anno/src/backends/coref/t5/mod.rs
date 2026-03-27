//! T5-based coreference resolution using ONNX Runtime.
//!
//! Experimental scaffold for seq2seq coreference.
//!
//! The intended approach treats coreference as a text-to-text transformation:
//!
//! ```text
//! Input:  "<m> Jensen </m> founded <m> NVIDIA </m>. <m> He </m> later led its AI division."
//! Output: "Jensen | 1 founded NVIDIA | 2. He | 1 later led its AI division."
//! ```
//!
//! The model learns to assign cluster IDs to mentions, enabling coreference
//! without explicit pairwise classification.
//!
//! **Status**: `T5Coref` loads ONNX artifacts, but `resolve()` currently uses a lightweight
//! heuristic fallback (it does not yet run a full encoder/decoder loop).
//!
//! # Architecture
//!
//! ```text
//! Text with Marked Mentions
//!         │
//!         ▼
//! ┌───────────────────┐
//! │   T5 Encoder      │
//! │   (ONNX)          │
//! └─────────┬─────────┘
//!           │
//!           ▼
//! ┌───────────────────┐
//! │   T5 Decoder      │
//! │   (Autoregressive)│
//! └─────────┬─────────┘
//!           │
//!           ▼
//! Text with Cluster IDs
//!         │
//!         ▼
//! ┌───────────────────┐
//! │  Parse Clusters   │
//! └───────────────────┘
//!         │
//!         ▼
//! CoreferenceCluster[]
//! ```
//!
//! # Model Export (One-Time Setup)
//!
//! Export a T5 coreference model to ONNX using Optimum:
//!
//! ```bash
//! pip install optimum[onnxruntime]
//! optimum-cli export onnx \
//!     --model "google/flan-t5-base" \
//!     --task text2text-generation-with-past \
//!     t5_coref_onnx/
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::backends::coref::t5::{T5Coref, T5CorefConfig};
//!
//! let coref = T5Coref::from_path("path/to/t5_coref_onnx", T5CorefConfig::default())?
//!     .with_heuristic_fallback();
//!
//! let text = "Sophie Wilson designed the ARM processor. She changed computing.";
//! let clusters = coref.resolve(text)?;
//!
//! // clusters[0] = { members: ["Sophie Wilson", "She"], canonical: "Sophie Wilson" }
//! ```
//!
//! # Research Background
//!
//! This approach is based on:
//! - Seq2seq coref: "Coreference Resolution as Query-based Span Prediction" (Wu et al.)
//! - FLAN-T5 fine-tuning for coreference tasks
//! - Entity-centric markup format for mention boundaries
//!
//! The seq2seq approach outperforms traditional pairwise classifiers on:
//! - OntoNotes 5.0 (coreference benchmark)
//! - GAP (gendered pronoun resolution benchmark)

// Note: This module is feature-gated via `#[cfg(feature = "onnx")]` in mod.rs

use crate::{Entity, Error, Result};

/// Return type for mention extraction: `(plain_text, [(mention_text, char_start, char_end)])`.
type MentionList = (String, Vec<(String, usize, usize)>);
use ndarray::{Array2, Array3};
use std::collections::HashMap;
use std::sync::Arc;

use ort::session::Session;
use tokenizers::Tokenizer;

// Use the canonical CorefCluster from the resolve module.
pub use super::resolve::CorefCluster;

/// Configuration for T5 coreference model.
#[derive(Debug, Clone)]
pub struct T5CorefConfig {
    /// Maximum input length (tokens)
    pub max_input_length: usize,
    /// Maximum output length (tokens)
    pub max_output_length: usize,
    /// Beam search width (1 = greedy)
    pub num_beams: usize,
    /// ONNX optimization level
    pub optimization_level: u8,
    /// Number of inference threads
    pub num_threads: usize,
}

impl Default for T5CorefConfig {
    fn default() -> Self {
        Self {
            max_input_length: 512,
            max_output_length: 512,
            num_beams: 1, // Greedy for speed
            optimization_level: 3,
            num_threads: 4,
        }
    }
}

/// T5-based coreference resolution.
///
/// Uses a seq2seq model to assign cluster IDs to marked mentions.
///
/// # Note
///
/// Currently uses a simplified rule-based fallback. Full seq2seq inference
/// is planned for a future release when encoder-decoder ONNX support matures.
pub struct T5Coref {
    /// Encoder ONNX session.
    encoder: std::sync::Mutex<Session>,
    /// Decoder ONNX session.
    decoder: std::sync::Mutex<Session>,
    /// HuggingFace tokenizer for input encoding and output decoding.
    tokenizer: Arc<Tokenizer>,
    /// Inference configuration.
    config: T5CorefConfig,
    /// Model path or HuggingFace model ID.
    model_path: String,
}

impl T5Coref {
    /// Create a new T5 coreference model from a local ONNX export.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to directory containing encoder.onnx and decoder_model.onnx
    pub fn from_path(model_path: &str, config: T5CorefConfig) -> Result<Self> {
        let encoder_path = format!("{}/encoder_model.onnx", model_path);
        let decoder_path = format!("{}/decoder_model.onnx", model_path);
        let tokenizer_path = format!("{}/tokenizer.json", model_path);

        // Check files exist
        if !std::path::Path::new(&encoder_path).exists() {
            return Err(Error::Retrieval(format!(
                "Encoder not found at {}. Export with: optimum-cli export onnx --model <model> --task text2text-generation-with-past {}",
                encoder_path, model_path
            )));
        }

        use crate::backends::hf_loader;

        let sess_config = hf_loader::OnnxSessionConfig {
            optimization_level: config.optimization_level,
            num_threads: config.num_threads,
            use_cpu_provider: true,
        };

        let encoder = hf_loader::create_onnx_session(
            std::path::Path::new(&encoder_path),
            sess_config.clone(),
        )?;
        let decoder =
            hf_loader::create_onnx_session(std::path::Path::new(&decoder_path), sess_config)?;
        let tokenizer = hf_loader::load_tokenizer(std::path::Path::new(&tokenizer_path))?;

        log::info!("[T5-Coref] Loaded model from {}", model_path);

        Ok(Self {
            encoder: std::sync::Mutex::new(encoder),
            decoder: std::sync::Mutex::new(decoder),
            tokenizer: Arc::new(tokenizer),
            config,
            model_path: model_path.to_string(),
        })
    }

    /// Create from HuggingFace model ID.
    ///
    /// Downloads ONNX-exported T5 model from HuggingFace Hub.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        Self::from_pretrained_with_config(model_id, T5CorefConfig::default())
    }

    /// Create from HuggingFace with custom config.
    pub fn from_pretrained_with_config(model_id: &str, config: T5CorefConfig) -> Result<Self> {
        use crate::backends::hf_loader;

        let api = hf_loader::hf_api()?;
        let repo = api.model(model_id.to_string());

        let encoder_path = hf_loader::download_model_file(
            &repo,
            &["encoder_model.onnx", "onnx/encoder_model.onnx"],
        )?;
        let decoder_path = hf_loader::download_model_file(
            &repo,
            &[
                "decoder_model.onnx",
                "onnx/decoder_model.onnx",
                "decoder_with_past_model.onnx",
            ],
        )?;
        let tokenizer_path = hf_loader::download_model_file(&repo, &["tokenizer.json"])?;

        let sess_config = hf_loader::OnnxSessionConfig {
            optimization_level: config.optimization_level,
            num_threads: config.num_threads,
            use_cpu_provider: true,
        };

        let encoder = hf_loader::create_onnx_session(&encoder_path, sess_config.clone())?;
        let decoder = hf_loader::create_onnx_session(&decoder_path, sess_config)?;
        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;

        log::info!("[T5-Coref] Loaded model from {}", model_id);

        Ok(Self {
            encoder: std::sync::Mutex::new(encoder),
            decoder: std::sync::Mutex::new(decoder),
            tokenizer: Arc::new(tokenizer),
            config,
            model_path: model_id.to_string(),
        })
    }

    /// Resolve coreference in text.
    ///
    /// Runs the T5 encoder-decoder loop when ONNX weights are available.
    /// Falls back to the rule-based heuristic if inference fails or produces
    /// no clusters (e.g. model not fine-tuned for coref, or GPU OOM).
    pub fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        match self.resolve_t5(text) {
            Ok(clusters) if !clusters.is_empty() => Ok(clusters),
            Ok(_) => {
                log::debug!("[T5-Coref] inference produced no clusters, using heuristic fallback");
                self.resolve_simple(text)
            }
            Err(e) => {
                log::warn!(
                    "[T5-Coref] inference failed ({}), using heuristic fallback",
                    e
                );
                self.resolve_simple(text)
            }
        }
    }

    // -------------------------------------------------------------------------
    // T5 encoder-decoder inference
    // -------------------------------------------------------------------------

    /// Full T5 inference path: mark → encode → greedy-decode → parse.
    fn resolve_t5(&self, text: &str) -> Result<Vec<CorefCluster>> {
        let marked = self.mark_mentions(text);
        let (input_ids, attention_mask) = self.tokenize_input(&marked)?;
        let (enc_hidden, enc_seq_len, hidden_size) =
            self.run_encoder(&input_ids, &attention_mask)?;
        let output_ids =
            self.greedy_decode(&enc_hidden, enc_seq_len, hidden_size, &attention_mask)?;
        let decoded = self.decode_tokens(&output_ids)?;
        Ok(self.parse_coref_output(&decoded))
    }

    /// Heuristically mark pronouns and capitalised tokens with `<m>…</m>` so the
    /// T5 model sees explicit mention boundaries.
    fn mark_mentions(&self, text: &str) -> String {
        mark_mentions_for_t5(text)
    }

    /// Tokenize `text` with the HuggingFace tokenizer.
    /// Returns `(input_ids, attention_mask)` as `i64` vecs, truncated to
    /// `config.max_input_length`.
    fn tokenize_input(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let mut enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Parse(format!("T5Coref tokenizer encode: {e}")))?;
        // Encoding::truncate returns () — not a Result.
        enc.truncate(
            self.config.max_input_length,
            0,
            tokenizers::TruncationDirection::Right,
        );
        let input_ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
        Ok((input_ids, attention_mask))
    }

    /// Run the T5 encoder.  Returns `(flat_hidden_states, seq_len, hidden_size)`.
    fn run_encoder(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
    ) -> Result<(Vec<f32>, usize, usize)> {
        let batch = 1usize;
        let seq_len = input_ids.len();

        let ids_arr = Array2::<i64>::from_shape_vec((batch, seq_len), input_ids.to_vec())
            .map_err(|e| Error::Parse(format!("encoder ids shape: {e}")))?;
        let mask_arr = Array2::<i64>::from_shape_vec((batch, seq_len), attention_mask.to_vec())
            .map_err(|e| Error::Parse(format!("encoder mask shape: {e}")))?;

        let ids_t = super::super::ort_compat::tensor_from_ndarray(ids_arr)
            .map_err(|e| Error::Parse(format!("encoder ids tensor: {e}")))?;
        let mask_t = super::super::ort_compat::tensor_from_ndarray(mask_arr)
            .map_err(|e| Error::Parse(format!("encoder mask tensor: {e}")))?;

        // Scope the mutex guard: `outputs` borrows from the session; extract owned
        // data before the guard drops.
        let (hidden_flat, hidden_size) = {
            let mut enc = self.encoder.lock().unwrap_or_else(|e| e.into_inner());
            let outputs = enc
                .run(ort::inputs![
                    "input_ids" => ids_t.into_dyn(),
                    "attention_mask" => mask_t.into_dyn(),
                ])
                .map_err(|e| Error::Parse(format!("T5Coref encoder run: {e}")))?;
            let hidden_val = outputs.get("last_hidden_state").ok_or_else(|| {
                Error::Parse(
                    "T5 encoder output 'last_hidden_state' not found; check ONNX export".into(),
                )
            })?;
            let (shape, data) = hidden_val
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Parse(format!("encoder extract tensor: {e}")))?;
            if shape.len() != 3 || shape[0] != 1 {
                return Err(Error::Parse(format!(
                    "T5 encoder: unexpected hidden-state shape {:?}",
                    shape
                )));
            }
            (data.to_vec(), shape[2] as usize)
        }; // enc guard drops here
        Ok((hidden_flat, seq_len, hidden_size))
    }

    /// Run one greedy decoder step and return the next token ID.
    ///
    /// The full `decoder_input_ids` sequence is fed each time (no KV-cache),
    /// which is O(n²) but correct and avoids managing past key-values.
    fn decoder_step(
        &self,
        encoder_hidden: &[f32],
        enc_seq_len: usize,
        hidden_size: usize,
        attention_mask: &[i64],
        decoder_input_ids: &[i64],
    ) -> Result<i64> {
        let batch = 1usize;
        let dec_len = decoder_input_ids.len();

        let enc_h = Array3::<f32>::from_shape_vec(
            (batch, enc_seq_len, hidden_size),
            encoder_hidden.to_vec(),
        )
        .map_err(|e| Error::Parse(format!("decoder enc_hidden shape: {e}")))?;
        let attn = Array2::<i64>::from_shape_vec((batch, enc_seq_len), attention_mask.to_vec())
            .map_err(|e| Error::Parse(format!("decoder attn shape: {e}")))?;
        let dec_ids = Array2::<i64>::from_shape_vec((batch, dec_len), decoder_input_ids.to_vec())
            .map_err(|e| Error::Parse(format!("decoder_ids shape: {e}")))?;

        let enc_h_t = super::super::ort_compat::tensor_from_ndarray(enc_h)
            .map_err(|e| Error::Parse(format!("enc_h tensor: {e}")))?;
        let attn_t = super::super::ort_compat::tensor_from_ndarray(attn)
            .map_err(|e| Error::Parse(format!("attn tensor: {e}")))?;
        let dec_ids_t = super::super::ort_compat::tensor_from_ndarray(dec_ids)
            .map_err(|e| Error::Parse(format!("dec_ids tensor: {e}")))?;

        // Scope the mutex guard: extract owned data before the guard drops.
        let next_token = {
            let mut dec = self.decoder.lock().unwrap_or_else(|e| e.into_inner());
            let outputs = dec
                .run(ort::inputs![
                    "encoder_hidden_states" => enc_h_t.into_dyn(),
                    "attention_mask"        => attn_t.into_dyn(),
                    "decoder_input_ids"     => dec_ids_t.into_dyn(),
                ])
                .map_err(|e| Error::Parse(format!("T5Coref decoder run: {e}")))?;
            let logits_val = outputs.get("logits").ok_or_else(|| {
                Error::Parse("T5 decoder output 'logits' not found; check ONNX export".into())
            })?;
            let (shape, logits_data) = logits_val
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Parse(format!("decoder logits extract: {e}")))?;
            // Expected shape: [1, dec_len, vocab_size]
            if shape.len() != 3 || shape[0] != 1 {
                return Err(Error::Parse(format!(
                    "T5 decoder: unexpected logits shape {:?}",
                    shape
                )));
            }
            let vocab_size = shape[2] as usize;
            let last_offset = (dec_len - 1) * vocab_size;
            let last_logits = &logits_data[last_offset..last_offset + vocab_size];
            last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap_or(1) // EOS as fallback
        }; // dec guard drops here
        Ok(next_token)
    }

    /// Greedy decode from encoder output.  Returns generated token IDs (excluding the
    /// leading pad/decoder-start token).
    fn greedy_decode(
        &self,
        encoder_hidden: &[f32],
        enc_seq_len: usize,
        hidden_size: usize,
        attention_mask: &[i64],
    ) -> Result<Vec<i64>> {
        // T5 uses pad token (0) as the decoder start token; EOS is 1.
        const T5_PAD: i64 = 0;
        const T5_EOS: i64 = 1;
        let mut generated = vec![T5_PAD];

        for _ in 0..self.config.max_output_length {
            let next = self.decoder_step(
                encoder_hidden,
                enc_seq_len,
                hidden_size,
                attention_mask,
                &generated,
            )?;
            if next == T5_EOS {
                break;
            }
            generated.push(next);
        }

        Ok(generated[1..].to_vec()) // drop the leading pad start token
    }

    /// Decode output token IDs to a string with the tokenizer.
    fn decode_tokens(&self, token_ids: &[i64]) -> Result<String> {
        let ids: Vec<u32> = token_ids.iter().map(|&x| x as u32).collect();
        self.tokenizer
            .decode(&ids, true)
            .map_err(|e| Error::Parse(format!("T5Coref decode_tokens: {e}")))
    }

    /// Parse T5 cluster-ID output format (`"word | N"`) into `CorefCluster`s.
    ///
    /// The expected output format is:
    /// ```text
    /// "Jensen | 1 founded NVIDIA | 2. He | 1 later led its AI division."
    /// ```
    /// where ` | N` immediately follows a mention token and assigns it to cluster `N`.
    ///
    /// Singletons (clusters with only one mention) are filtered out.
    fn parse_coref_output(&self, decoded: &str) -> Vec<CorefCluster> {
        parse_t5_coref_output(decoded)
    }

    /// Resolve coreference with pre-marked mentions.
    ///
    /// Expects mentions marked with `<m>` and `</m>` tags:
    /// `"<m> Sophie Wilson </m> designed ARM. <m> She </m> changed computing."`
    /// Resolve coreference with pre-marked mentions.
    ///
    /// Expects mentions marked with `<m>` and `</m>` tags:
    /// `"<m> Sophie Wilson </m> designed ARM. <m> She </m> changed computing."`
    ///
    /// Runs T5 inference directly on the marked text (skipping the auto-marking
    /// step used by `resolve()`).  Falls back to similarity-based clustering when
    /// inference fails or produces no clusters.
    pub fn resolve_marked(&self, marked_text: &str) -> Result<Vec<CorefCluster>> {
        let (plain_text, mentions) = self.extract_mentions(marked_text)?;
        if mentions.is_empty() {
            return Ok(vec![]);
        }
        // The text is already marked — feed it directly to T5 without re-marking.
        match self.resolve_t5_raw(marked_text) {
            Ok(clusters) if !clusters.is_empty() => Ok(clusters),
            Ok(_) => self.cluster_mentions(&plain_text, &mentions),
            Err(e) => {
                log::warn!(
                    "[T5-Coref] resolve_marked inference failed ({}), using fallback",
                    e
                );
                self.cluster_mentions(&plain_text, &mentions)
            }
        }
    }

    /// Resolve coreference for a set of entities from NER.
    ///
    /// Reconstructs `<m>…</m>` markers from entity spans, then runs T5 inference.
    /// Falls back to similarity-based clustering when inference fails.
    pub fn resolve_entities(&self, text: &str, entities: &[Entity]) -> Result<Vec<CorefCluster>> {
        if entities.is_empty() {
            return Ok(vec![]);
        }

        // Rebuild marked text from entity spans so T5 sees explicit mention boundaries.
        let marked = self.mark_entity_spans(text, entities);
        match self.resolve_t5_raw(&marked) {
            Ok(clusters) if !clusters.is_empty() => Ok(clusters),
            Ok(_) => {
                let mentions: Vec<(String, usize, usize)> = entities
                    .iter()
                    .map(|e| (e.text.clone(), e.start(), e.end()))
                    .collect();
                self.cluster_mentions(text, &mentions)
            }
            Err(e) => {
                log::warn!(
                    "[T5-Coref] resolve_entities inference failed ({}), using fallback",
                    e
                );
                let mentions: Vec<(String, usize, usize)> = entities
                    .iter()
                    .map(|e| (e.text.clone(), e.start(), e.end()))
                    .collect();
                self.cluster_mentions(text, &mentions)
            }
        }
    }

    /// Run T5 on pre-marked text (already has `<m>…</m>` tags).
    ///
    /// This is the shared inner path for [`resolve_marked`] and [`resolve_entities`];
    /// unlike [`resolve_t5`] it does **not** call `mark_mentions`.
    fn resolve_t5_raw(&self, marked_text: &str) -> Result<Vec<CorefCluster>> {
        let (input_ids, attention_mask) = self.tokenize_input(marked_text)?;
        let (enc_hidden, enc_seq_len, hidden_size) =
            self.run_encoder(&input_ids, &attention_mask)?;
        let output_ids =
            self.greedy_decode(&enc_hidden, enc_seq_len, hidden_size, &attention_mask)?;
        let decoded = self.decode_tokens(&output_ids)?;
        Ok(self.parse_coref_output(&decoded))
    }

    /// Reconstruct a `<m>…</m>`-marked string from entity spans.
    ///
    /// Entities are sorted by start offset; overlapping spans are skipped.
    fn mark_entity_spans(&self, text: &str, entities: &[Entity]) -> String {
        let chars: Vec<char> = text.chars().collect();
        let char_len = chars.len();

        let mut sorted: Vec<&Entity> = entities.iter().collect();
        sorted.sort_by_key(|e| e.start());

        let mut out = String::with_capacity(text.len() + entities.len() * 10);
        let mut cursor = 0usize; // char offset

        for e in &sorted {
            if e.start() >= e.end() || e.start() < cursor || e.end() > char_len {
                continue;
            }
            // Text before this entity
            for &ch in &chars[cursor..e.start()] {
                out.push(ch);
            }
            out.push_str("<m> ");
            for &ch in &chars[e.start()..e.end()] {
                out.push(ch);
            }
            out.push_str(" </m>");
            cursor = e.end();
        }

        // Remaining text
        for &ch in &chars[cursor..] {
            out.push(ch);
        }
        out
    }

    /// Simple rule-based coreference (fallback).
    fn resolve_simple(&self, text: &str) -> Result<Vec<CorefCluster>> {
        // Simple heuristic: find pronouns and link to nearest compatible noun
        let pronouns = ["he", "she", "they", "it", "his", "her", "their", "its"];

        let words: Vec<(String, usize, usize)> = {
            let mut result = Vec::new();
            let mut pos = 0;
            for word in text.split_whitespace() {
                if let Some(start) = text[pos..].find(word) {
                    let abs_start = pos + start;
                    result.push((word.to_string(), abs_start, abs_start + word.len()));
                    pos = abs_start + word.len();
                }
            }
            result
        };

        // Find potential antecedents (capitalized words, likely names)
        let antecedents: Vec<&(String, usize, usize)> = words
            .iter()
            .filter(|(w, _, _)| {
                w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && !pronouns.contains(&w.to_lowercase().as_str())
            })
            .collect();

        // Find pronouns
        let pronoun_mentions: Vec<&(String, usize, usize)> = words
            .iter()
            .filter(|(w, _, _)| pronouns.contains(&w.to_lowercase().as_str()))
            .collect();

        // Build clusters
        let mut clusters: Vec<CorefCluster> = Vec::new();
        let mut assigned: HashMap<usize, u32> = HashMap::new();

        for (ant_text, ant_start, ant_end) in &antecedents {
            // Check if already assigned
            if assigned.contains_key(ant_start) {
                continue;
            }

            let cluster_id = clusters.len() as u32;
            let mut mentions = vec![ant_text.clone()];
            let mut spans = vec![(*ant_start, *ant_end)];

            assigned.insert(*ant_start, cluster_id);

            // Find pronouns after this antecedent that could refer to it
            for (pro_text, pro_start, pro_end) in &pronoun_mentions {
                if *pro_start > *ant_end && !assigned.contains_key(pro_start) {
                    // Check gender compatibility (simplified)
                    let compatible = match pro_text.to_lowercase().as_str() {
                        "he" | "him" | "his" => true, // Could be anyone
                        "she" | "her" | "hers" => true,
                        "they" | "them" | "their" | "theirs" => true,
                        "it" | "its" => true,
                        _ => true,
                    };

                    if compatible {
                        mentions.push(pro_text.clone());
                        spans.push((*pro_start, *pro_end));
                        assigned.insert(*pro_start, cluster_id);
                        break; // Only link nearest pronoun
                    }
                }
            }

            if mentions.len() > 1 {
                clusters.push(CorefCluster {
                    id: cluster_id,
                    canonical: ant_text.clone(),
                    mentions,
                    spans,
                });
            }
        }

        Ok(clusters)
    }

    fn extract_mentions(&self, marked_text: &str) -> Result<MentionList> {
        extract_t5_mentions(marked_text)
    }

    /// Cluster mentions by similarity.
    fn cluster_mentions(
        &self,
        _text: &str,
        mentions: &[(String, usize, usize)],
    ) -> Result<Vec<CorefCluster>> {
        // Simple clustering: exact match + substring match + pronoun resolution
        let mut clusters: Vec<CorefCluster> = Vec::new();
        let mut assigned: HashMap<usize, u32> = HashMap::new();

        // English-only pronoun list for T5 coref clustering heuristic.
        let pronouns = [
            "he", "she", "they", "it", "him", "her", "them", "his", "hers", "their", "its",
        ];

        for (i, (text_i, start_i, end_i)) in mentions.iter().enumerate() {
            if assigned.contains_key(&i) {
                continue;
            }

            let lower_i = text_i.to_lowercase();
            let is_pronoun_i = pronouns.contains(&lower_i.as_str());

            if is_pronoun_i {
                // Find nearest preceding non-pronoun
                for j in (0..i).rev() {
                    let (text_j, _, _) = &mentions[j];
                    let lower_j = text_j.to_lowercase();
                    if !pronouns.contains(&lower_j.as_str()) {
                        if let Some(&cluster_id) = assigned.get(&j) {
                            assigned.insert(i, cluster_id);
                            clusters[cluster_id as usize].mentions.push(text_i.clone());
                            clusters[cluster_id as usize].spans.push((*start_i, *end_i));
                        }
                        break;
                    }
                }
                continue;
            }

            // Start new cluster
            let cluster_id = clusters.len() as u32;
            let mut cluster_mentions = vec![text_i.clone()];
            let mut cluster_spans = vec![(*start_i, *end_i)];
            assigned.insert(i, cluster_id);

            // Find matches
            for (j, (text_j, start_j, end_j)) in mentions.iter().enumerate().skip(i + 1) {
                if assigned.contains_key(&j) {
                    continue;
                }

                let lower_j = text_j.to_lowercase();

                // Exact match
                let matches = lower_i == lower_j
                    // Substring match
                    || lower_i.contains(&lower_j)
                    || lower_j.contains(&lower_i)
                    // Last word match (surname)
                    || {
                        let last_i = lower_i.split_whitespace().last();
                        let last_j = lower_j.split_whitespace().last();
                        last_i.is_some() && last_i == last_j && last_i.map(|w| w.len() > 2).unwrap_or(false)
                    };

                if matches {
                    cluster_mentions.push(text_j.clone());
                    cluster_spans.push((*start_j, *end_j));
                    assigned.insert(j, cluster_id);
                }
            }

            // Determine canonical (longest mention)
            let canonical = cluster_mentions
                .iter()
                .max_by_key(|m| m.len())
                .cloned()
                .unwrap_or_else(|| text_i.clone());

            clusters.push(CorefCluster {
                id: cluster_id,
                mentions: cluster_mentions,
                spans: cluster_spans,
                canonical,
            });
        }

        // Filter to only multi-mention clusters
        let multi_clusters: Vec<CorefCluster> = clusters
            .into_iter()
            .filter(|c| c.mentions.len() > 1)
            .collect();

        Ok(multi_clusters)
    }

    /// Get model path.
    pub fn model_path(&self) -> &str {
        &self.model_path
    }
}

// =============================================================================
// Free-function helpers (pure parsing — no ONNX, directly testable)
// =============================================================================

/// Heuristically mark pronouns and capitalised tokens in `text` with `<m>…</m>` tags.
///
/// This is the same logic used by `T5Coref::mark_mentions` and is exposed as a free
/// function so it can be tested and reused without an ONNX session.
pub fn mark_mentions_for_t5(text: &str) -> String {
    // English-only pronoun list for T5 mention marking.
    const PRONOUNS: &[&str] = &[
        "he", "she", "they", "it", "him", "her", "them", "his", "hers", "their", "its",
    ];
    let mut out = String::with_capacity(text.len() + 64);
    for (i, word) in text.split_whitespace().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let lower = word
            .trim_matches(|c: char| !c.is_alphabetic())
            .to_lowercase();
        let is_pronoun = PRONOUNS.contains(&lower.as_str());
        let is_cap = word
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);
        if is_pronoun || is_cap {
            out.push_str("<m> ");
            out.push_str(word);
            out.push_str(" </m>");
        } else {
            out.push_str(word);
        }
    }
    out
}

/// Parse T5 cluster-ID output format (`"word | N"`) into [`CorefCluster`]s.
///
/// Singletons are filtered out.  This is the same logic as
/// `T5Coref::parse_coref_output` and is exposed as a free function for testing.
pub fn parse_t5_coref_output(decoded: &str) -> Vec<CorefCluster> {
    let mut clusters: HashMap<u32, CorefCluster> = HashMap::new();
    let tokens: Vec<&str> = decoded.split_whitespace().collect();
    let mut offset: usize = 0;
    let mut i = 0;

    while i < tokens.len() {
        let tok = tokens[i];
        let is_pipe = tokens.get(i + 1).map(|&t| t == "|").unwrap_or(false);
        let cluster_id: Option<u32> = if is_pipe {
            tokens
                .get(i + 2)
                .and_then(|t| t.trim_matches(|c: char| !c.is_ascii_digit()).parse().ok())
        } else {
            None
        };

        if let Some(cid) = cluster_id {
            let mention = tok.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if !mention.is_empty() {
                let start = offset;
                let end = offset + mention.len();
                let entry = clusters.entry(cid).or_insert_with(|| CorefCluster {
                    id: cid,
                    mentions: Vec::new(),
                    spans: Vec::new(),
                    canonical: String::new(),
                });
                entry.mentions.push(mention);
                entry.spans.push((start, end));
            }
            offset += tok.len() + 1;
            i += 3;
            continue;
        }

        offset += tok.len() + 1;
        i += 1;
    }

    let mut result: Vec<CorefCluster> = clusters
        .into_values()
        .filter(|c| c.mentions.len() > 1)
        .collect();
    for c in &mut result {
        c.canonical = c
            .mentions
            .iter()
            .max_by_key(|m| m.len())
            .cloned()
            .unwrap_or_default();
    }
    result.sort_by_key(|c| c.id);
    result
}

/// Extract `<m>…</m>` spans from `marked_text`, returning `(plain_text, mentions)`.
///
/// Each mention is `(text, char_start, char_end)` in the plain text.
/// This is the same logic as `T5Coref::extract_mentions` and is exposed as a
/// free function for testing.
pub fn extract_t5_mentions(marked_text: &str) -> Result<MentionList> {
    let mut plain_text = String::new();
    let mut mentions = Vec::new();
    let mut offset = 0;

    let mut remaining = marked_text;
    while !remaining.is_empty() {
        if let Some(start_pos) = remaining.find("<m>") {
            plain_text.push_str(&remaining[..start_pos]);
            offset += start_pos;

            let after_start = &remaining[start_pos + 3..];
            if let Some(end_pos) = after_start.find("</m>") {
                let mention_text = after_start[..end_pos].trim();
                let mention_start = offset;
                plain_text.push_str(mention_text);
                let mention_end = offset + mention_text.len();
                offset = mention_end;

                mentions.push((mention_text.to_string(), mention_start, mention_end));
                remaining = &after_start[end_pos + 4..];
            } else {
                plain_text.push_str(remaining);
                break;
            }
        } else {
            plain_text.push_str(remaining);
            break;
        }
    }

    Ok((plain_text, mentions))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
