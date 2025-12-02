//! Session pool for high-throughput ONNX inference.
//!
//! A single `Mutex<Session>` serializes all inference requests. For high-throughput
//! scenarios (batch processing, web servers), a pool of sessions enables parallel
//! inference across multiple threads.
//!
//! # Architecture
//!
//! ```text
//! Without Pool (serialized):
//! ──────────────────────────
//! Thread 1 → [lock] → [inference] → [unlock]
//!                                    Thread 2 → [lock] → [inference] → [unlock]
//!                                                                       Thread 3 → ...
//!
//! With Pool (parallel):
//! ─────────────────────
//! Thread 1 → [session 1] → [inference] → [return to pool]
//! Thread 2 → [session 2] → [inference] → [return to pool]
//! Thread 3 → [session 3] → [inference] → [return to pool]
//! ```
//!
//! # Memory vs Throughput Trade-off
//!
//! Each session holds a copy of model weights in memory. A pool of N sessions
//! uses ~N× the memory of a single session. Choose pool size based on:
//!
//! - **CPU cores**: More sessions than cores wastes memory
//! - **Model size**: Large models (>500MB) may limit pool size
//! - **Request pattern**: Bursty traffic benefits from larger pools
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::backends::session_pool::{SessionPool, PoolConfig};
//!
//! // Create pool with 4 sessions
//! let pool = SessionPool::new(
//!     "onnx-community/gliner_small-v2.1",
//!     PoolConfig::with_size(4),
//! )?;
//!
//! // Use from multiple threads
//! let entities = pool.extract("John works at Apple", &["person", "organization"], 0.5)?;
//! ```

#![cfg(feature = "session-pool")]

use crate::{Entity, Error, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::Arc;

#[cfg(feature = "onnx")]
use {
    hf_hub::api::sync::Api,
    ndarray::{Array2, Array3},
    ort::{execution_providers::CPUExecutionProvider, session::Session, value::Tensor},
    std::path::PathBuf,
    tokenizers::Tokenizer,
};

/// Configuration for the session pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of sessions in the pool
    pub pool_size: usize,
    /// Timeout for acquiring a session (milliseconds)
    pub acquire_timeout_ms: u64,
    /// Whether to use quantized models if available
    pub prefer_quantized: bool,
    /// ONNX graph optimization level (1-3)
    pub optimization_level: u8,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            pool_size: num_cpus(),
            acquire_timeout_ms: 5000,
            prefer_quantized: true,
            optimization_level: 3,
        }
    }
}

impl PoolConfig {
    /// Create config with specific pool size.
    #[must_use]
    pub fn with_size(size: usize) -> Self {
        Self {
            pool_size: size,
            ..Default::default()
        }
    }

    /// Set timeout for acquiring sessions.
    #[must_use]
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.acquire_timeout_ms = timeout_ms;
        self
    }

    /// Prefer quantized models for faster inference.
    #[must_use]
    pub fn prefer_quantized(mut self, prefer: bool) -> Self {
        self.prefer_quantized = prefer;
        self
    }
}

/// Get number of CPUs (fallback to 4).
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

/// A pooled ONNX session that returns to the pool on drop.
#[cfg(feature = "onnx")]
pub struct PooledSession {
    session: Option<Session>,
    return_tx: Sender<Session>,
}

#[cfg(feature = "onnx")]
impl Drop for PooledSession {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            // Return session to pool (ignore error if pool is closed)
            let _ = self.return_tx.send(session);
        }
    }
}

#[cfg(feature = "onnx")]
impl std::ops::Deref for PooledSession {
    type Target = Session;

    fn deref(&self) -> &Self::Target {
        self.session.as_ref().expect("Session already returned")
    }
}

#[cfg(feature = "onnx")]
impl std::ops::DerefMut for PooledSession {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.session.as_mut().expect("Session already returned")
    }
}

/// Pool of ONNX sessions for high-throughput inference.
///
/// Each session can process requests independently, enabling true parallel
/// inference across multiple threads.
#[cfg(feature = "onnx")]
pub struct SessionPool {
    /// Channel to return sessions
    tx: Sender<Session>,
    /// Channel to acquire sessions
    rx: Receiver<Session>,
    /// Shared tokenizer (thread-safe)
    tokenizer: Arc<Tokenizer>,
    /// Model name
    model_name: String,
    /// Pool configuration
    config: PoolConfig,
}

#[cfg(feature = "onnx")]
impl SessionPool {
    /// Create a new session pool.
    ///
    /// Downloads the model and creates `pool_size` independent ONNX sessions.
    ///
    /// # Arguments
    ///
    /// * `model_name` - HuggingFace model ID (e.g., "onnx-community/gliner_small-v2.1")
    /// * `config` - Pool configuration
    pub fn new(model_name: &str, config: PoolConfig) -> Result<Self> {
        let api = Api::new()
            .map_err(|e| Error::Retrieval(format!("HuggingFace API init failed: {}", e)))?;

        let repo = api.model(model_name.to_string());

        // Download model (try quantized first if preferred)
        let model_path = if config.prefer_quantized {
            repo.get("model_quantized.onnx")
                .or_else(|_| repo.get("onnx/model_quantized.onnx"))
                .or_else(|_| repo.get("onnx/model.onnx"))
                .or_else(|_| repo.get("model.onnx"))
        } else {
            repo.get("onnx/model.onnx")
                .or_else(|_| repo.get("model.onnx"))
        }
        .map_err(|e| Error::Retrieval(format!("Model download failed: {}", e)))?;

        // Download tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| Error::Retrieval(format!("Tokenizer download failed: {}", e)))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Retrieval(format!("Tokenizer load failed: {}", e)))?;

        // Create pool channels
        let (tx, rx) = bounded(config.pool_size);

        // Create sessions
        for i in 0..config.pool_size {
            let session = Self::create_session(&model_path, &config)?;
            tx.send(session)
                .map_err(|_| Error::Retrieval(format!("Failed to add session {} to pool", i)))?;
        }

        log::info!(
            "[SessionPool] Created {} sessions for {}",
            config.pool_size,
            model_name
        );

        Ok(Self {
            tx,
            rx,
            tokenizer: Arc::new(tokenizer),
            model_name: model_name.to_string(),
            config,
        })
    }

    /// Create a single ONNX session.
    fn create_session(model_path: &PathBuf, config: &PoolConfig) -> Result<Session> {
        use ort::session::builder::GraphOptimizationLevel;

        let opt_level = match config.optimization_level {
            1 => GraphOptimizationLevel::Level1,
            2 => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::Level3,
        };

        Session::builder()
            .map_err(|e| Error::Retrieval(format!("Session builder: {}", e)))?
            .with_optimization_level(opt_level)
            .map_err(|e| Error::Retrieval(format!("Optimization level: {}", e)))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| Error::Retrieval(format!("Execution providers: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| Error::Retrieval(format!("Session load: {}", e)))
    }

    /// Acquire a session from the pool.
    ///
    /// Blocks until a session is available or timeout is reached.
    pub fn acquire(&self) -> Result<PooledSession> {
        let timeout = std::time::Duration::from_millis(self.config.acquire_timeout_ms);

        let session = self
            .rx
            .recv_timeout(timeout)
            .map_err(|_| Error::Retrieval("Session pool timeout".to_string()))?;

        Ok(PooledSession {
            session: Some(session),
            return_tx: self.tx.clone(),
        })
    }

    /// Get shared tokenizer reference.
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    /// Get pool size.
    pub fn pool_size(&self) -> usize {
        self.config.pool_size
    }

    /// Get number of available sessions.
    pub fn available(&self) -> usize {
        self.rx.len()
    }

    /// Get model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

// =============================================================================
// GLiNER-specific Pool
// =============================================================================

/// Special token IDs for GLiNER models.
#[cfg(feature = "onnx")]
const TOKEN_START: u32 = 1;
#[cfg(feature = "onnx")]
const TOKEN_END: u32 = 2;
#[cfg(feature = "onnx")]
const TOKEN_ENT: u32 = 128002;
#[cfg(feature = "onnx")]
const TOKEN_SEP: u32 = 128003;
#[cfg(feature = "onnx")]
const MAX_SPAN_WIDTH: usize = 12;

/// GLiNER session pool for zero-shot NER.
///
/// Optimized for GLiNER's bi-encoder architecture with pre-allocated
/// span tensors and efficient entity type encoding.
#[cfg(feature = "onnx")]
pub struct GLiNERPool {
    pool: SessionPool,
}

#[cfg(feature = "onnx")]
impl GLiNERPool {
    /// Create a new GLiNER pool.
    pub fn new(model_name: &str, config: PoolConfig) -> Result<Self> {
        let pool = SessionPool::new(model_name, config)?;
        Ok(Self { pool })
    }

    /// Extract entities using zero-shot NER.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to extract from
    /// * `entity_types` - Entity type labels (e.g., ["person", "organization"])
    /// * `threshold` - Confidence threshold (0.0-1.0)
    pub fn extract(
        &self,
        text: &str,
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        if text.is_empty() || entity_types.is_empty() {
            return Ok(vec![]);
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec![]);
        }

        // Encode prompt
        let (input_ids, attention_mask, words_mask, _text_len, _entity_count) =
            self.encode_prompt(&words, entity_types)?;

        // Generate span tensors
        let (span_idx, span_mask) = Self::make_span_tensors(words.len());

        // Acquire session
        let mut session = self.pool.acquire()?;

        // Build tensors
        let batch_size = 1;
        let seq_len = input_ids.len();
        // Use checked_mul to prevent overflow (same pattern as gliner2.rs:2388)
        let num_spans = words.len().checked_mul(MAX_SPAN_WIDTH).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Span count overflow: {} words * {} MAX_SPAN_WIDTH",
                words.len(),
                MAX_SPAN_WIDTH
            ))
        })?;

        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), input_ids)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let attention_mask_array = Array2::from_shape_vec((batch_size, seq_len), attention_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let words_mask_array = Array2::from_shape_vec((batch_size, seq_len), words_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let text_lengths_array = Array2::from_shape_vec((batch_size, 1), vec![words.len() as i64])
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let span_idx_array = Array3::from_shape_vec((batch_size, num_spans, 2), span_idx)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;
        let span_mask_array = Array2::from_shape_vec((batch_size, num_spans), span_mask)
            .map_err(|e| Error::Parse(format!("Array error: {}", e)))?;

        let input_ids_t = Tensor::from_array(input_ids_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let attention_mask_t = Tensor::from_array(attention_mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let words_mask_t = Tensor::from_array(words_mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let text_lengths_t = Tensor::from_array(text_lengths_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let span_idx_t = Tensor::from_array(span_idx_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;
        let span_mask_t = Tensor::from_array(span_mask_array)
            .map_err(|e| Error::Parse(format!("Tensor error: {}", e)))?;

        // Run inference
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_t.into_dyn(),
                "attention_mask" => attention_mask_t.into_dyn(),
                "words_mask" => words_mask_t.into_dyn(),
                "text_lengths" => text_lengths_t.into_dyn(),
                "span_idx" => span_idx_t.into_dyn(),
                "span_mask" => span_mask_t.into_dyn(),
            ])
            .map_err(|e| Error::Parse(format!("ONNX inference failed: {}", e)))?;

        // Decode output
        self.decode_output(&outputs, text, &words, entity_types, threshold)
    }

    /// Encode prompt following GLiNER pattern.
    fn encode_prompt(
        &self,
        words: &[&str],
        entity_types: &[&str],
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>, i64, usize)> {
        let tokenizer = self.pool.tokenizer();

        let mut input_ids: Vec<i64> = Vec::new();
        let mut word_mask: Vec<i64> = Vec::new();

        // Start token
        input_ids.push(TOKEN_START as i64);
        word_mask.push(0);

        // Entity types
        for entity_type in entity_types {
            input_ids.push(TOKEN_ENT as i64);
            word_mask.push(0);

            let encoding = tokenizer
                .encode(entity_type.to_string(), false)
                .map_err(|e| Error::Parse(format!("Tokenizer error: {}", e)))?;
            for token_id in encoding.get_ids() {
                input_ids.push(*token_id as i64);
                word_mask.push(0);
            }
        }

        // Separator
        input_ids.push(TOKEN_SEP as i64);
        word_mask.push(0);

        // Text words
        let mut word_id: i64 = 0;
        for word in words {
            let encoding = tokenizer
                .encode(word.to_string(), false)
                .map_err(|e| Error::Parse(format!("Tokenizer error: {}", e)))?;

            word_id += 1;

            for (token_idx, token_id) in encoding.get_ids().iter().enumerate() {
                input_ids.push(*token_id as i64);
                word_mask.push(if token_idx == 0 { word_id } else { 0 });
            }
        }

        // End token
        input_ids.push(TOKEN_END as i64);
        word_mask.push(0);

        let seq_len = input_ids.len();
        let attention_mask: Vec<i64> = vec![1; seq_len];

        Ok((
            input_ids,
            attention_mask,
            word_mask,
            word_id,
            entity_types.len(),
        ))
    }

    /// Generate span tensors.
    fn make_span_tensors(num_words: usize) -> (Vec<i64>, Vec<bool>) {
        // Use checked_mul to prevent overflow (same pattern as gliner2.rs:2388)
        let num_spans = num_words.checked_mul(MAX_SPAN_WIDTH).unwrap_or_else(|| {
            log::warn!(
                "Span count overflow: {} words * {} MAX_SPAN_WIDTH, using max",
                num_words,
                MAX_SPAN_WIDTH
            );
            usize::MAX
        });
        // Check for overflow in num_spans * 2
        let span_idx_len = num_spans.checked_mul(2).unwrap_or_else(|| {
            log::warn!(
                "Span idx length overflow: {} spans * 2, using max",
                num_spans
            );
            usize::MAX
        });
        let mut span_idx: Vec<i64> = vec![0; span_idx_len];
        let mut span_mask: Vec<bool> = vec![false; num_spans];

        for start in 0..num_words {
            let remaining = num_words - start;
            let actual_max = MAX_SPAN_WIDTH.min(remaining);

            for width in 0..actual_max {
                // Check for overflow in dim calculation (same pattern as nuner.rs:399)
                let dim = match start.checked_mul(MAX_SPAN_WIDTH) {
                    Some(v) => match v.checked_add(width) {
                        Some(d) => d,
                        None => {
                            log::warn!(
                                "Dim calculation overflow: {} * {} + {}, skipping span",
                                start,
                                MAX_SPAN_WIDTH,
                                width
                            );
                            continue;
                        }
                    },
                    None => {
                        log::warn!(
                            "Dim calculation overflow: {} * {}, skipping span",
                            start,
                            MAX_SPAN_WIDTH
                        );
                        continue;
                    }
                };
                // Check bounds before array access (dim * 2 could overflow or exceed span_idx_len)
                if let Some(dim2) = dim.checked_mul(2) {
                    if dim2 + 1 < span_idx_len && dim < num_spans {
                        span_idx[dim2] = start as i64;
                        span_idx[dim2 + 1] = (start + width) as i64;
                        span_mask[dim] = true;
                    } else {
                        log::warn!(
                            "Span idx access out of bounds: dim={}, dim*2={}, span_idx_len={}, num_spans={}, skipping",
                            dim, dim2, span_idx_len, num_spans
                        );
                    }
                } else {
                    log::warn!("Dim * 2 overflow: dim={}, skipping span", dim);
                }
            }
        }

        (span_idx, span_mask)
    }

    /// Decode model output to entities.
    fn decode_output(
        &self,
        outputs: &ort::session::SessionOutputs,
        text: &str,
        words: &[&str],
        entity_types: &[&str],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        let output = outputs
            .iter()
            .next()
            .map(|(_, v)| v)
            .ok_or_else(|| Error::Parse("No output".to_string()))?;

        let (_, data_slice) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Parse(format!("Extract tensor: {}", e)))?;
        let output_data: Vec<f32> = data_slice.to_vec();

        let shape: Vec<i64> = match output.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
            _ => return Err(Error::Parse("Not a tensor".to_string())),
        };

        if output_data.is_empty() || shape.iter().any(|&d| d == 0) {
            return Ok(vec![]);
        }

        let mut entities = Vec::new();
        let num_words = words.len();

        if shape.len() == 4 && shape[0] == 1 {
            let out_num_words = shape[1] as usize;
            let out_max_width = shape[2] as usize;
            let num_classes = shape[3] as usize;

            if num_classes == 0 {
                return Ok(vec![]);
            }

            for word_idx in 0..out_num_words.min(num_words) {
                for width in 0..out_max_width.min(MAX_SPAN_WIDTH) {
                    let end_word = word_idx + width;
                    if end_word >= num_words {
                        continue;
                    }

                    for class_idx in 0..num_classes.min(entity_types.len()) {
                        let idx = (word_idx * out_max_width * num_classes)
                            + (width * num_classes)
                            + class_idx;

                        if idx < output_data.len() {
                            let logit = output_data[idx];
                            let score = 1.0 / (1.0 + (-logit).exp());

                            if score >= threshold {
                                let span_text = words[word_idx..=end_word].join(" ");
                                let (start, end) =
                                    Self::word_offsets(text, words, word_idx, end_word);

                                let type_str = entity_types.get(class_idx).unwrap_or(&"OTHER");
                                let entity_type = Self::map_type(type_str);

                                entities.push(Entity::new(
                                    span_text,
                                    entity_type,
                                    start,
                                    end,
                                    score as f64,
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Performance: Use unstable sort (we don't need stable sort here)
        // Sort and deduplicate
        entities.sort_unstable_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| b.end.cmp(&a.end))
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        entities.dedup_by(|a, b| a.start == b.start && a.end == b.end);

        Ok(entities)
    }

    /// Convert word indices to character offsets.
    fn word_offsets(text: &str, words: &[&str], start: usize, end: usize) -> (usize, usize) {
        let mut pos = 0;
        let mut start_char = 0;
        let mut end_char = text.len();

        for (idx, word) in words.iter().enumerate() {
            if let Some(found) = text[pos..].find(word) {
                let word_start = pos + found;
                let word_end = word_start + word.len();

                if idx == start {
                    start_char = word_start;
                }
                if idx == end {
                    end_char = word_end;
                    break;
                }
                pos = word_end;
            }
        }

        (start_char, end_char)
    }

    /// Map string to EntityType.
    fn map_type(type_str: &str) -> crate::EntityType {
        use crate::EntityType;

        match type_str.to_lowercase().as_str() {
            "person" | "per" => EntityType::Person,
            "organization" | "org" => EntityType::Organization,
            "location" | "loc" | "gpe" => EntityType::Location,
            "date" | "time" => EntityType::Date,
            "money" | "currency" => EntityType::Money,
            "percent" | "percentage" => EntityType::Percent,
            other => EntityType::Other(other.to_string()),
        }
    }

    /// Get underlying pool.
    pub fn pool(&self) -> &SessionPool {
        &self.pool
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert!(config.pool_size > 0);
        assert!(config.acquire_timeout_ms > 0);
    }

    #[test]
    fn test_pool_config_builder() {
        let config = PoolConfig::with_size(8)
            .with_timeout(10000)
            .prefer_quantized(false);

        assert_eq!(config.pool_size, 8);
        assert_eq!(config.acquire_timeout_ms, 10000);
        assert!(!config.prefer_quantized);
    }

    // Integration tests require model download
    #[test]
    #[ignore = "Requires model download"]
    fn test_session_pool_creation() {
        let config = PoolConfig::with_size(2);
        let pool = SessionPool::new("onnx-community/gliner_small-v2.1", config);
        assert!(pool.is_ok());

        let pool = pool.unwrap();
        assert_eq!(pool.pool_size(), 2);
        assert_eq!(pool.available(), 2);
    }
}
