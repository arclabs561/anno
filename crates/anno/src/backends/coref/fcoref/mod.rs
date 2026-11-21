//! F-coref: fast neural coreference resolution.
//!
//! Rust implementation of the f-coref model (Otmazgin et al., AACL 2022).
//! Uses ONNX for the DistilRoBERTa encoder and pure Rust (ndarray) for the
//! scorer heads.
//!
//! # Architecture
//!
//! ```text
//! Text
//!   |
//!   v
//! Tokenizer (DistilRoBERTa)
//!   |
//!   v
//! Encoder (ONNX, sequential segments) -> document hidden_states [T, 768]
//!   |
//!   v
//! Mention Scorer (ndarray) -> top-k mention spans
//!   |
//!   v
//! Antecedent Scorer (ndarray) -> best antecedent per mention
//!   |
//!   v
//! Union-Find Clustering -> CorefCluster[]
//! ```
//!
//! # Model Export
//!
//! Use the Python export script to obtain the ONNX encoder and scorer weights:
//!
//! ```bash
//! uv run scripts/export_fcoref.py --output-dir fcoref_onnx
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use anno::backends::coref::fcoref::FCoref;
//!
//! let coref = FCoref::from_path("fcoref_onnx")?;
//! let clusters = coref.resolve("John went to the store. He bought milk.")?;
//! // clusters[0] = { mentions: ["John", "He"], canonical: "John" }
//! # Ok(())
//! # }
//! ```

pub(crate) mod clustering;
pub(crate) mod scoring;

use std::sync::Arc;

use ndarray::Array2;
use ort::session::Session;
use tokenizers::Tokenizer;

use super::resolve::CorefCluster;
use crate::offset::SpanConverter;
use crate::{Error, Result};
use clustering::MentionSpan;
use scoring::ScorerWeights;

/// Configuration for f-coref model loading.
#[derive(Debug, Clone)]
pub struct FCorefConfig {
    /// Maximum span width for mention candidates.
    pub max_span_length: usize,
    /// Fraction of tokens to keep as candidate mentions.
    pub top_lambda: f32,
    /// maximum encoder segment length, not a whole-document limit.
    pub max_segment_len: usize,
    /// maximum document tokens, including special tokens. default: `Some(512)`.
    ///
    /// larger documents return an error, never truncated annotations. raise this
    /// budget to encode more segments, or use `None` to remove the document cap.
    /// document-level scoring uses quadratic memory; bound untrusted inputs.
    pub max_doc_tokens: Option<usize>,
    /// ONNX optimization level (1-3, default 3).
    pub optimization_level: u8,
    /// ONNX encoder threads (0 = auto). does not control the Rust scorer.
    pub num_threads: usize,
}

impl Default for FCorefConfig {
    fn default() -> Self {
        Self {
            max_span_length: 30,
            top_lambda: 0.25,
            max_segment_len: 512,
            max_doc_tokens: Some(512),
            optimization_level: 3,
            num_threads: 4,
        }
    }
}

impl FCorefConfig {
    fn validate_limits(&self) -> Result<()> {
        if self.max_segment_len == 0 || self.max_doc_tokens == Some(0) {
            return Err(Error::InvalidInput(
                "coreference segment and document limits must be positive".into(),
            ));
        }
        Ok(())
    }
}

/// F-coref neural coreference resolver.
///
/// Uses a DistilRoBERTa encoder (ONNX) with learned scorer heads (ndarray/safetensors)
/// to perform fast, accurate within-document coreference resolution.
pub struct FCoref {
    encoder: std::sync::Mutex<Session>,
    tokenizer: Arc<Tokenizer>,
    scorer: ScorerWeights,
    config: FCorefConfig,
    model_path: String,
}

impl FCoref {
    /// Load from a local directory containing exported artifacts.
    ///
    /// Expects:
    /// - `encoder.onnx` (or `encoder_quantized.onnx`)
    /// - `scorer_weights.safetensors`
    /// - `tokenizer.json`
    /// - `config.json`
    pub fn from_path(model_path: &str) -> Result<Self> {
        Self::from_path_with_config(model_path, FCorefConfig::default())
    }

    /// Load with custom configuration.
    pub fn from_path_with_config(model_path: &str, mut config: FCorefConfig) -> Result<Self> {
        config.validate_limits()?;
        let base = std::path::Path::new(model_path);

        // Load config.json if present (override defaults)
        let config_path = base.join("config.json");
        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| Error::Retrieval(format!("Failed to read config.json: {}", e)))?;
            let cfg: serde_json::Value = serde_json::from_str(&config_str)
                .map_err(|e| Error::Parse(format!("Failed to parse config.json: {}", e)))?;
            if let Some(head) = cfg.get("coref_head") {
                if let Some(v) = head.get("max_span_length").and_then(|v| v.as_u64()) {
                    config.max_span_length = v as usize;
                }
                if let Some(v) = head.get("top_lambda").and_then(|v| v.as_f64()) {
                    config.top_lambda = v as f32;
                }
                if let Some(v) = head.get("max_segment_len").and_then(|v| v.as_u64()) {
                    config.max_segment_len = v as usize;
                }
            }
        }

        config.validate_limits()?;

        // Load encoder ONNX session (prefer quantized)
        let encoder_path = if base.join("encoder_quantized.onnx").exists() {
            log::info!("[f-coref] Using quantized encoder");
            base.join("encoder_quantized.onnx")
        } else {
            base.join("encoder.onnx")
        };

        if !encoder_path.exists() {
            return Err(Error::Retrieval(format!(
                "Encoder not found at {}. Run: uv run scripts/export_fcoref.py --output-dir {}",
                encoder_path.display(),
                model_path
            )));
        }

        use crate::backends::hf_loader;

        let session = hf_loader::create_onnx_session(
            &encoder_path,
            hf_loader::OnnxSessionConfig {
                optimization_level: config.optimization_level,
                num_threads: config.num_threads,
                use_cpu_provider: false,
                ..Default::default()
            },
        )?;

        let tokenizer_path = base.join("tokenizer.json");
        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;

        // Load scorer weights
        let weights_path = base.join("scorer_weights.safetensors");
        let scorer = ScorerWeights::from_safetensors(&weights_path)?;

        log::info!("[f-coref] Loaded model from {}", model_path);

        Ok(Self {
            encoder: std::sync::Mutex::new(session),
            tokenizer: Arc::new(tokenizer),
            scorer,
            config,
            model_path: model_path.to_string(),
        })
    }

    /// Load from HuggingFace Hub.
    ///
    /// Downloads pre-exported ONNX + safetensors artifacts from the specified model ID.
    /// The model repo must contain `encoder.onnx`, `scorer_weights.safetensors`,
    /// `tokenizer.json`, and `config.json`.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        Self::from_pretrained_with_config(model_id, FCorefConfig::default())
    }

    /// Load from HuggingFace with custom config.
    pub fn from_pretrained_with_config(model_id: &str, mut config: FCorefConfig) -> Result<Self> {
        config.validate_limits()?;
        use crate::backends::hf_loader;

        let api = hf_loader::hf_api()?;
        let repo = api.model(model_id.to_string());

        // Helper to download a file, returning Ok(None) if not found
        let try_get = |name: &str| -> Result<Option<std::path::PathBuf>> {
            match hf_loader::download_model_file(&repo, &[name]) {
                Ok(p) => Ok(Some(p)),
                Err(_) => Ok(None),
            }
        };

        // Download required files
        let weights_path =
            hf_loader::download_model_file(&repo, &["scorer_weights.safetensors"])
                .map_err(|e| Error::Retrieval(format!("scorer_weights download: {}", e)))?;
        let tokenizer_path = hf_loader::download_model_file(&repo, &["tokenizer.json"])
            .map_err(|e| Error::Retrieval(format!("tokenizer download: {}", e)))?;

        // Prefer quantized encoder if available
        let encoder_path = if let Some(q) = try_get("encoder_quantized.onnx")? {
            log::info!("[f-coref] Using quantized encoder from {}", model_id);
            q
        } else {
            hf_loader::download_model_file(&repo, &["encoder.onnx"])
                .map_err(|e| Error::Retrieval(format!("encoder.onnx download: {}", e)))?
        };

        // Parse config.json if available (override defaults)
        if let Some(config_path) = try_get("config.json")? {
            if let Ok(config_str) = std::fs::read_to_string(&config_path) {
                if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&config_str) {
                    if let Some(head) = cfg.get("coref_head") {
                        if let Some(v) = head.get("max_span_length").and_then(|v| v.as_u64()) {
                            config.max_span_length = v as usize;
                        }
                        if let Some(v) = head.get("top_lambda").and_then(|v| v.as_f64()) {
                            config.top_lambda = v as f32;
                        }
                        if let Some(v) = head.get("max_segment_len").and_then(|v| v.as_u64()) {
                            config.max_segment_len = v as usize;
                        }
                    }
                }
            }
        }

        config.validate_limits()?;

        // Load encoder session
        let session = hf_loader::create_onnx_session(
            &encoder_path,
            hf_loader::OnnxSessionConfig {
                optimization_level: config.optimization_level,
                num_threads: config.num_threads,
                use_cpu_provider: false,
                ..Default::default()
            },
        )?;

        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;

        // Load scorer weights
        let scorer = ScorerWeights::from_safetensors(&weights_path)?;

        log::info!("[f-coref] Loaded model from {}", model_id);

        Ok(Self {
            encoder: std::sync::Mutex::new(session),
            tokenizer: Arc::new(tokenizer),
            scorer,
            config,
            model_path: model_id.to_string(),
        })
    }

    /// Resolve coreference in text.
    ///
    /// Returns clusters of co-referring mentions with character offsets.
    ///
    /// the full document is encoded in sequential, non-overlapping segments,
    /// then scored together. raising `max_doc_tokens` does not change the
    /// encoder's segment size or add independent coreference windows.
    /// inputs exceeding the document budget or using tokenizer truncation fail
    /// before encoder inference, rather than silently dropping their tail.
    pub fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // 1. Tokenize
        let encoding = encode_input(&self.tokenizer, text, &self.config)?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        // 2. Encode all segments before document-level mention/antecedent scoring.
        let hidden = encode_segments(
            &input_ids,
            &attention_mask,
            self.config.max_segment_len,
            |ids, mask| self.run_encoder(ids, mask),
        )?;

        // 3. Score mentions over the complete document.
        let mentions_result = scoring::score_mentions(
            &hidden,
            &self.scorer,
            self.config.max_span_length,
            self.config.top_lambda,
        );

        if mentions_result.top_k_starts.is_empty() {
            return Ok(vec![]);
        }

        // 4. Score antecedents
        let antecedents = scoring::score_antecedents(
            &mentions_result.top_k_starts,
            &mentions_result.top_k_ends,
            &mentions_result.top_k_logits,
            &mentions_result.start_coref_reps,
            &mentions_result.end_coref_reps,
            &self.scorer,
        );

        // 5. Map token indices to character offsets
        let offsets = encoding.get_offsets();
        let span_converter = SpanConverter::new(text);

        let mentions: Vec<MentionSpan> = mentions_result
            .top_k_starts
            .iter()
            .zip(mentions_result.top_k_ends.iter())
            .filter_map(|(&ts, &te)| {
                if ts >= offsets.len() || te >= offsets.len() {
                    return None;
                }
                let (byte_start, _) = offsets[ts];
                let (_, byte_end) = offsets[te];
                if byte_start >= byte_end || byte_end > text.len() {
                    return None;
                }
                let mention_text = text.get(byte_start..byte_end)?.trim();
                if mention_text.is_empty() {
                    return None;
                }
                Some(MentionSpan {
                    token_start: ts,
                    token_end: te,
                    char_start: span_converter.byte_to_char(byte_start),
                    char_end: span_converter.byte_to_char(byte_end),
                    text: mention_text.to_string(),
                })
            })
            .collect();

        // 6. Build clusters
        // Re-index antecedents to match the filtered mentions
        let original_count = mentions_result.top_k_starts.len();
        let mut index_map = vec![None; original_count];
        for (new_idx, mention) in mentions.iter().enumerate() {
            // Find original index by matching token positions
            for (old_idx, (&os, &oe)) in mentions_result
                .top_k_starts
                .iter()
                .zip(mentions_result.top_k_ends.iter())
                .enumerate()
            {
                if os == mention.token_start && oe == mention.token_end {
                    index_map[old_idx] = Some(new_idx);
                    break;
                }
            }
        }

        let filtered_antecedents: Vec<usize> = mentions
            .iter()
            .enumerate()
            .map(|(new_i, mention)| {
                // Find original index
                let old_i = mentions_result
                    .top_k_starts
                    .iter()
                    .zip(mentions_result.top_k_ends.iter())
                    .position(|(&os, &oe)| os == mention.token_start && oe == mention.token_end)
                    .unwrap_or(new_i);

                let old_ante = antecedents.get(old_i).copied().unwrap_or(old_i);
                // Map to new index
                index_map.get(old_ante).and_then(|&x| x).unwrap_or(new_i) // null if antecedent was filtered out
            })
            .collect();

        Ok(clustering::build_clusters(&mentions, &filtered_antecedents))
    }

    /// Run the DistilRoBERTa encoder and return hidden states.
    fn run_encoder(&self, input_ids: &[i64], attention_mask: &[i64]) -> Result<Array2<f32>> {
        let seq_len = input_ids.len();

        let ids_arr = Array2::<i64>::from_shape_vec((1, seq_len), input_ids.to_vec())
            .map_err(|e| Error::Parse(format!("ids shape: {}", e)))?;
        let mask_arr = Array2::<i64>::from_shape_vec((1, seq_len), attention_mask.to_vec())
            .map_err(|e| Error::Parse(format!("mask shape: {}", e)))?;

        let ids_t = super::super::ort_compat::tensor_from_ndarray(ids_arr)
            .map_err(|e| Error::Parse(format!("ids tensor: {}", e)))?;
        let mask_t = super::super::ort_compat::tensor_from_ndarray(mask_arr)
            .map_err(|e| Error::Parse(format!("mask tensor: {}", e)))?;

        let hidden_flat = {
            let mut session = self.encoder.lock().unwrap_or_else(|e| e.into_inner());
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => ids_t.into_dyn(),
                    "attention_mask" => mask_t.into_dyn(),
                ])
                .map_err(|e| Error::Parse(format!("Encoder run: {}", e)))?;

            let hidden_val = outputs.get("last_hidden_state").ok_or_else(|| {
                Error::Parse("Encoder output 'last_hidden_state' not found".into())
            })?;
            let (shape, data) = hidden_val
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Parse(format!("Extract tensor: {}", e)))?;

            if shape.len() != 3 || shape[0] != 1 {
                return Err(Error::Parse(format!(
                    "Unexpected hidden shape: {:?}",
                    shape
                )));
            }
            let hidden_size = shape[2] as usize;
            Array2::from_shape_vec((seq_len, hidden_size), data.to_vec())
                .map_err(|e| Error::Parse(format!("Hidden reshape: {}", e)))?
        };

        Ok(hidden_flat)
    }

    /// Get model path.
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// Get configuration.
    pub fn config(&self) -> &FCorefConfig {
        &self.config
    }
}

fn encode_input(
    tokenizer: &Tokenizer,
    text: &str,
    config: &FCorefConfig,
) -> Result<tokenizers::Encoding> {
    config.validate_limits()?;
    if tokenizer.get_truncation().is_some() {
        return Err(Error::InvalidInput(
            "coreference tokenizer must not truncate input".into(),
        ));
    }
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| Error::Parse(format!("Tokenizer encode: {e}")))?;
    if let Some(limit) = config.max_doc_tokens {
        if encoding.len() > limit {
            return Err(Error::InvalidInput(format!(
                "coreference document has {} tokens, exceeding max_doc_tokens {limit}",
                encoding.len()
            )));
        }
    }
    Ok(encoding)
}

fn encode_segments(
    ids: &[i64],
    mask: &[i64],
    segment_len: usize,
    mut encode: impl FnMut(&[i64], &[i64]) -> Result<Array2<f32>>,
) -> Result<Array2<f32>> {
    if segment_len == 0 || ids.is_empty() || ids.len() != mask.len() {
        return Err(Error::InvalidInput("invalid encoder segment input".into()));
    }
    let first_end = segment_len.min(ids.len());
    let first = encode(&ids[..first_end], &mask[..first_end])?;
    let width = first.ncols();
    if first.nrows() != first_end || width == 0 {
        return Err(Error::Parse(
            "encoder returned an invalid segment shape".into(),
        ));
    }
    if first_end == ids.len() {
        return Ok(first);
    }
    let mut hidden = Array2::zeros((ids.len(), width));
    hidden
        .slice_mut(ndarray::s![..first_end, ..])
        .assign(&first);
    drop(first);
    // offsets still refer to the original tokenization: do not insert new
    // special tokens or independently resolve/reconcile segment clusters.
    for start in (first_end..ids.len()).step_by(segment_len) {
        let end = (start + segment_len).min(ids.len());
        let part = encode(&ids[start..end], &mask[start..end])?;
        if part.dim() != (end - start, width) {
            return Err(Error::Parse(
                "encoder returned an inconsistent segment shape".into(),
            ));
        }
        hidden.slice_mut(ndarray::s![start..end, ..]).assign(&part);
    }
    Ok(hidden)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FCorefConfig::default();
        assert_eq!(config.max_span_length, 30);
        assert!((config.top_lambda - 0.25).abs() < 1e-6);
        assert_eq!(config.max_segment_len, 512);
        assert_eq!(config.max_doc_tokens, Some(512));
        assert_eq!(config.optimization_level, 3);
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_config_custom() {
        let config = FCorefConfig {
            max_span_length: 15,
            top_lambda: 0.4,
            max_segment_len: 256,
            max_doc_tokens: Some(4096),
            optimization_level: 1,
            num_threads: 2,
        };
        assert_eq!(config.max_span_length, 15);
        assert!((config.top_lambda - 0.4).abs() < 1e-6);
        assert_eq!(config.max_doc_tokens, Some(4096));
    }

    fn test_tokenizer() -> Tokenizer {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
        use tokenizers::processors::template::TemplateProcessing;
        let model = WordLevel::builder()
            .vocab(
                [("[UNK]".into(), 0), ("<s>".into(), 1), ("</s>".into(), 2)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<s> $A </s>")
                .unwrap()
                .special_tokens(vec![("<s>", 1), ("</s>", 2)])
                .build()
                .unwrap(),
        ));
        tokenizer
    }

    #[test]
    fn document_budget_counts_special_tokens_and_can_be_disabled() {
        let tokenizer = test_tokenizer();
        let config = FCorefConfig {
            max_doc_tokens: Some(4),
            ..Default::default()
        };
        assert_eq!(
            encode_input(&tokenizer, "one two", &config).unwrap().len(),
            4
        );
        assert!(matches!(
            encode_input(&tokenizer, "one two three", &config),
            Err(Error::InvalidInput(_))
        ));
        let text = "word ".repeat(600);
        assert!(encode_input(&tokenizer, &text, &FCorefConfig::default()).is_err());
        let larger = FCorefConfig {
            max_doc_tokens: Some(1024),
            ..Default::default()
        };
        assert_eq!(encode_input(&tokenizer, &text, &larger).unwrap().len(), 602);
        let unlimited = FCorefConfig {
            max_doc_tokens: None,
            ..Default::default()
        };
        assert_eq!(
            encode_input(&tokenizer, &text, &unlimited).unwrap().len(),
            602
        );
    }

    #[test]
    fn tokenizer_truncation_is_rejected_instead_of_hiding_the_tail() {
        let mut tokenizer = test_tokenizer();
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 4,
                ..Default::default()
            }))
            .unwrap();
        let config = FCorefConfig {
            max_doc_tokens: None,
            ..Default::default()
        };
        assert!(matches!(
            encode_input(&tokenizer, "one two three four", &config),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn zero_limits_fail_before_loading_or_downloading_artifacts() {
        let configs = [
            FCorefConfig {
                max_doc_tokens: Some(0),
                ..Default::default()
            },
            FCorefConfig {
                max_segment_len: 0,
                ..Default::default()
            },
        ];
        for config in configs {
            assert!(matches!(
                FCoref::from_path_with_config("missing-model", config.clone()),
                Err(Error::InvalidInput(_))
            ));
            assert!(matches!(
                FCoref::from_pretrained_with_config("unused-model", config),
                Err(Error::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn segments_reconstruct_global_token_order_without_inserting_special_tokens() {
        let ids = [101, 10, 20, 30, 40, 50, 60, 102];
        let mask = [1, 1, 1, 0, 1, 1, 1, 1];
        let mut calls = Vec::new();
        let hidden = encode_segments(&ids, &mask, 3, |part, attention| {
            calls.push((part.to_vec(), attention.to_vec()));
            Ok(Array2::from_shape_fn((part.len(), 2), |(row, col)| {
                if col == 0 {
                    part[row] as f32
                } else {
                    attention[row] as f32
                }
            }))
        })
        .unwrap();
        assert_eq!(
            calls.iter().map(|(ids, _)| ids.len()).collect::<Vec<_>>(),
            vec![3, 3, 2]
        );
        assert_eq!(
            calls
                .iter()
                .flat_map(|(ids, _)| ids.iter().copied())
                .collect::<Vec<_>>(),
            ids
        );
        assert_eq!(
            calls
                .iter()
                .flat_map(|(_, mask)| mask.iter().copied())
                .collect::<Vec<_>>(),
            mask
        );
        assert_eq!(hidden.dim(), (8, 2));
        for row in 0..ids.len() {
            assert_eq!(hidden[[row, 0]], ids[row] as f32);
            assert_eq!(hidden[[row, 1]], mask[row] as f32);
        }
    }

    #[test]
    fn short_input_uses_one_encoder_call() {
        let mut calls = 0;
        let hidden = encode_segments(&[1, 2], &[1, 1], 512, |ids, _| {
            calls += 1;
            Ok(Array2::zeros((ids.len(), 3)))
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(hidden.dim(), (2, 3));
    }

    #[test]
    fn encoder_failures_stop_without_a_partial_document_result() {
        let mut calls = 0;
        let result = encode_segments(&[1; 10], &[1; 10], 4, |ids, _| {
            calls += 1;
            if calls == 2 {
                return Err(Error::Parse("encoder failed".into()));
            }
            Ok(Array2::zeros((ids.len(), 2)))
        });
        assert!(matches!(result, Err(Error::Parse(_))));
        assert_eq!(calls, 2);
    }

    #[test]
    fn malformed_segment_inputs_and_outputs_are_rejected() {
        for (ids, mask, width) in [
            (&[][..], &[][..], 4),
            (&[1][..], &[][..], 4),
            (&[1][..], &[1][..], 0),
        ] {
            assert!(encode_segments(ids, mask, width, |_, _| panic!("must not encode")).is_err());
        }
        assert!(encode_segments(&[1; 4], &[1; 4], 2, |_, _| Ok(Array2::zeros((1, 2)))).is_err());
        let mut calls = 0;
        assert!(encode_segments(&[1; 4], &[1; 4], 2, |ids, _| {
            calls += 1;
            Ok(Array2::zeros((ids.len(), calls)))
        })
        .is_err());
    }

    // Integration tests require model download -- mark as #[ignore].
    // Run with: cargo test -p anno --features onnx -- fcoref --ignored

    fn model_dir() -> String {
        let manifest = env!("CARGO_MANIFEST_DIR");
        format!("{}/fcoref_onnx", manifest)
    }

    #[test]
    #[ignore]
    fn test_fcoref_basic_resolution() {
        let coref = FCoref::from_path(&model_dir())
            .expect("Model not found. Run: uv run scripts/export_fcoref.py");
        let clusters = coref
            .resolve("John went to the store. He bought milk.")
            .unwrap();
        // Should find John-He cluster
        assert!(
            !clusters.is_empty(),
            "Expected at least one coreference cluster"
        );
        let has_john_he = clusters.iter().any(|c| {
            c.mentions.iter().any(|m| m.contains("John")) && c.mentions.iter().any(|m| m == "He")
        });
        assert!(has_john_he, "Expected John-He cluster, got: {:?}", clusters);
    }

    #[test]
    #[ignore]
    fn test_fcoref_no_coreference() {
        let coref = FCoref::from_path(&model_dir())
            .expect("Model not found. Run: uv run scripts/export_fcoref.py");
        let clusters = coref.resolve("The weather is nice today.").unwrap();
        // No pronouns referring to named entities
        assert!(
            clusters.is_empty(),
            "Expected no clusters for non-referential text"
        );
    }

    #[test]
    #[ignore]
    fn test_fcoref_long_chain() {
        let coref = FCoref::from_path(&model_dir())
            .expect("Model not found. Run: uv run scripts/export_fcoref.py");
        let text = "Marie Curie was born in Warsaw. She studied in Paris. \
                     She discovered radium. She won two Nobel Prizes.";
        let clusters = coref.resolve(text).unwrap();
        // Should find a cluster linking Marie Curie with She mentions
        let curie_cluster = clusters.iter().find(|c| {
            c.mentions
                .iter()
                .any(|m| m.contains("Marie") || m.contains("Curie"))
        });
        assert!(
            curie_cluster.is_some(),
            "Expected Marie Curie cluster, got: {:?}",
            clusters
        );
        if let Some(c) = curie_cluster {
            assert!(
                c.mentions.len() >= 3,
                "Expected 3+ mentions in Marie Curie cluster, got {}",
                c.mentions.len()
            );
        }
    }
}
