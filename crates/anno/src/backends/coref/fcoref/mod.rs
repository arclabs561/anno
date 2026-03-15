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
//! Encoder (ONNX) -> hidden_states [1, T, 768]
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
    /// Maximum input length in tokens.
    pub max_segment_len: usize,
    /// ONNX optimization level (1-3, default 3).
    pub optimization_level: u8,
    /// Number of threads for inference (0 = auto).
    pub num_threads: usize,
}

impl Default for FCorefConfig {
    fn default() -> Self {
        Self {
            max_span_length: 30,
            top_lambda: 0.25,
            max_segment_len: 512,
            optimization_level: 3,
            num_threads: 4,
        }
    }
}

/// F-coref neural coreference resolver.
///
/// Uses a DistilRoBERTa encoder (ONNX) with learned scorer heads (ndarray/safetensors)
/// to perform fast, accurate within-document coreference resolution.
pub struct FCoref {
    encoder: crate::sync::Mutex<Session>,
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
            },
        )?;

        let tokenizer_path = base.join("tokenizer.json");
        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;

        // Load scorer weights
        let weights_path = base.join("scorer_weights.safetensors");
        let scorer = ScorerWeights::from_safetensors(&weights_path)?;

        log::info!("[f-coref] Loaded model from {}", model_path);

        Ok(Self {
            encoder: crate::sync::Mutex::new(session),
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
        use crate::backends::hf_loader;

        let api = hf_loader::hf_api()?;
        let repo = api.model(model_id.to_string());

        // Helper to download a file, returning Ok(None) if not found
        let try_get = |name: &str| -> Result<Option<std::path::PathBuf>> {
            match repo.get(name) {
                Ok(p) => Ok(Some(p)),
                Err(_) => Ok(None),
            }
        };

        // Download required files
        let weights_path = repo
            .get("scorer_weights.safetensors")
            .map_err(|e| Error::Retrieval(format!("scorer_weights download: {}", e)))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Retrieval(format!("tokenizer download: {}", e)))?;

        // Prefer quantized encoder if available
        let encoder_path = if let Some(q) = try_get("encoder_quantized.onnx")? {
            log::info!("[f-coref] Using quantized encoder from {}", model_id);
            q
        } else {
            repo.get("encoder.onnx")
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

        // Load encoder session
        let session = hf_loader::create_onnx_session(
            &encoder_path,
            hf_loader::OnnxSessionConfig {
                optimization_level: config.optimization_level,
                num_threads: config.num_threads,
                use_cpu_provider: false,
            },
        )?;

        let tokenizer = hf_loader::load_tokenizer(&tokenizer_path)?;

        // Load scorer weights
        let scorer = ScorerWeights::from_safetensors(&weights_path)?;

        log::info!("[f-coref] Loaded model from {}", model_id);

        Ok(Self {
            encoder: crate::sync::Mutex::new(session),
            tokenizer: Arc::new(tokenizer),
            scorer,
            config,
            model_path: model_id.to_string(),
        })
    }

    /// Resolve coreference in text.
    ///
    /// Returns clusters of co-referring mentions with character offsets.
    pub fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // 1. Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Parse(format!("Tokenizer encode: {}", e)))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        let seq_len = input_ids.len().min(self.config.max_segment_len);
        let input_ids = &input_ids[..seq_len];
        let attention_mask = &attention_mask[..seq_len];

        // 2. Run encoder
        let hidden = self.run_encoder(input_ids, attention_mask)?;

        // 3. Score mentions (skip special tokens: positions 1..seq_len-1)
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
            let mut session = crate::sync::lock(&self.encoder);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FCorefConfig::default();
        assert_eq!(config.max_span_length, 30);
        assert!((config.top_lambda - 0.25).abs() < 1e-6);
        assert_eq!(config.max_segment_len, 512);
        assert_eq!(config.optimization_level, 3);
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_config_custom() {
        let config = FCorefConfig {
            max_span_length: 15,
            top_lambda: 0.4,
            max_segment_len: 256,
            optimization_level: 1,
            num_threads: 2,
        };
        assert_eq!(config.max_span_length, 15);
        assert!((config.top_lambda - 0.4).abs() < 1e-6);
    }

    // Integration tests require model download -- mark as #[ignore].
    // Run with: cargo test -p anno-lib --features onnx -- fcoref --ignored

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
