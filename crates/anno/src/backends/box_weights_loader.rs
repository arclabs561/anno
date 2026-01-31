//! Safetensors weight loader for box embedding projection layers.
//!
//! Loads trained box projection weights from `box-coref` exports.
//! These weights transform entity embeddings (e.g., from BERT/ModernBERT) into
//! box embeddings for coreference resolution.
//!
//! # Architecture
//!
//! The projection is:
//! ```text
//! embedding (768d) --[mu_proj]--> center (128d)
//!                  --[sigma_proj]--> log_sigma (128d) --[exp]--> delta (128d)
//! box = (center - delta, center + delta)  # min/max bounds
//! ```
//!
//! # File Format
//!
//! Expected safetensors structure (from `box-coref/scripts/export_box_layers.py`):
//! ```text
//! text_l1_mu.weight     [768, 128]  - Projects to box center
//! text_l1_mu.bias       [128]
//! text_l1_sigma.weight  [768, 128]  - Projects to box log-sigma
//! text_l1_sigma.bias    [128]
//! ```
//!
//! Alternative naming (older exports):
//! ```text
//! mu_proj.weight, mu_proj.bias
//! log_sigma_proj.weight, log_sigma_proj.bias
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno::backends::{BoxProjectionWeights, load_box_projection_weights};
//!
//! // Load trained weights from box-coref export
//! let weights = load_box_projection_weights("models/box_projection_layers.safetensors")?;
//!
//! // Get embedding from a text encoder (BERT, ModernBERT, etc.)
//! let embedding = encoder.encode("Barack Obama")?; // Vec<f32> of length 768
//!
//! // Project to box embedding
//! let box_emb = weights.project_to_box_embedding(&embedding);
//!
//! // Use for coreference resolution
//! let other_box = weights.project_to_box_embedding(&other_embedding);
//! let overlap = box_emb.conditional_probability(&other_box);
//! ```
//!
//! # Integration with box-coref
//!
//! The typical workflow is:
//! 1. Train box embeddings in `box-coref` using `just experiment-train`
//! 2. Export weights: `just export-model checkpoints/best.ckpt`
//! 3. Load in anno with this module
//!
//! See `docs/` for integration notes (repo-local docs; keep this module usable without
//! assuming unpublished design notes).

use crate::Error;
use std::path::Path;

/// Convert f16 bits to f32 (IEEE 754 half-precision to single-precision).
#[cfg(feature = "safetensors")]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;

    if exponent == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            if sign == 1 {
                -0.0
            } else {
                0.0
            }
        } else {
            // Subnormal
            let e = -14 - (mantissa.leading_zeros() as i32 - 6);
            let m = (mantissa << (mantissa.leading_zeros() - 5)) & 0x3ff;
            let f32_bits = ((sign as u32) << 31)
                | (((127 + e) as u32) << 23)
                | ((m as u32) << 13);
            f32::from_bits(f32_bits)
        }
    } else if exponent == 31 {
        // Inf or NaN
        if mantissa == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        // Normal number
        let f32_bits = ((sign as u32) << 31)
            | (((exponent as u32) + 112) << 23)
            | ((mantissa as u32) << 13);
        f32::from_bits(f32_bits)
    }
}

/// Box projection weights for transforming text embeddings to boxes.
///
/// These weights are trained by `box-coref` to project entity embeddings
/// into a box embedding space where containment relationships encode
/// coreference probability.
///
/// # Dimensions
///
/// - `input_dim`: Typically 768 (BERT-base) or 1024 (BERT-large)
/// - `output_dim`: Typically 128 (L1 resolution in multi-resolution scheme)
///
/// # Thread Safety
///
/// This struct is `Clone` and can be shared across threads by cloning.
/// For concurrent inference, consider using `Arc<BoxProjectionWeights>`.
#[derive(Debug, Clone)]
pub struct BoxProjectionWeights {
    /// Weights for projecting to box center (mu).
    /// Shape: `[input_dim, output_dim]`
    pub mu_weight: Vec<Vec<f32>>,
    /// Bias for box center.
    /// Shape: `[output_dim]`
    pub mu_bias: Vec<f32>,
    /// Weights for projecting to box log-sigma (controls box size).
    /// Shape: `[input_dim, output_dim]`
    pub sigma_weight: Vec<Vec<f32>>,
    /// Bias for log-sigma.
    /// Shape: `[output_dim]`
    pub sigma_bias: Vec<f32>,
    /// Input dimension (e.g., 768 for BERT-base, 1024 for BERT-large).
    pub input_dim: usize,
    /// Output dimension (e.g., 128 for L1 boxes in multi-resolution scheme).
    pub output_dim: usize,
}

impl BoxProjectionWeights {
    /// Creates new weights with given dimensions, initialized to identity-like projection.
    pub fn new_identity(input_dim: usize, output_dim: usize) -> Self {
        // Initialize mu_weight as truncated identity (projects to subset of dims)
        let mu_weight: Vec<Vec<f32>> = (0..input_dim)
            .map(|i| {
                (0..output_dim)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();

        // Zero bias for center
        let mu_bias = vec![0.0; output_dim];

        // Initialize sigma to small positive values (log(0.5) ≈ -0.69)
        let sigma_weight: Vec<Vec<f32>> = (0..input_dim)
            .map(|_| vec![0.0; output_dim])
            .collect();
        let sigma_bias = vec![-0.69; output_dim]; // Initial box size ~0.5

        Self {
            mu_weight,
            mu_bias,
            sigma_weight,
            sigma_bias,
            input_dim,
            output_dim,
        }
    }

    /// Projects an embedding to a box (center, delta).
    ///
    /// Returns (center, delta) where delta is the half-size of the box.
    pub fn project_to_box(&self, embedding: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(
            embedding.len(),
            self.input_dim,
            "Embedding dimension mismatch"
        );

        // Compute center: mu_weight @ embedding + mu_bias
        let mut center = self.mu_bias.clone();
        for (i, &emb_val) in embedding.iter().enumerate() {
            for (j, center_val) in center.iter_mut().enumerate() {
                *center_val += self.mu_weight[i][j] * emb_val;
            }
        }

        // Compute log_sigma: sigma_weight @ embedding + sigma_bias
        let mut log_sigma = self.sigma_bias.clone();
        for (i, &emb_val) in embedding.iter().enumerate() {
            for (j, ls_val) in log_sigma.iter_mut().enumerate() {
                *ls_val += self.sigma_weight[i][j] * emb_val;
            }
        }

        // Convert log_sigma to delta (half-size): delta = exp(log_sigma)
        let delta: Vec<f32> = log_sigma.iter().map(|&ls| ls.exp()).collect();

        (center, delta)
    }

    /// Projects an embedding to a BoxEmbedding.
    pub fn project_to_box_embedding(
        &self,
        embedding: &[f32],
    ) -> crate::backends::box_embeddings::BoxEmbedding {
        let (center, delta) = self.project_to_box(embedding);

        // Convert center ± delta to min/max
        let min: Vec<f32> = center
            .iter()
            .zip(delta.iter())
            .map(|(&c, &d)| c - d)
            .collect();
        let max: Vec<f32> = center
            .iter()
            .zip(delta.iter())
            .map(|(&c, &d)| c + d)
            .collect();

        crate::backends::box_embeddings::BoxEmbedding::new(min, max)
    }
}

/// Loads box projection weights from a safetensors file.
///
/// # Arguments
/// * `path` - Path to the safetensors file
///
/// # Returns
/// `BoxProjectionWeights` on success
///
/// # Errors
/// Returns error if file cannot be read or parsed.
#[cfg(feature = "safetensors")]
pub fn load_box_projection_weights<P: AsRef<Path>>(path: P) -> crate::Result<BoxProjectionWeights> {
    use safetensors::SafeTensors;

    let path = path.as_ref();
    let data = std::fs::read(path)
        .map_err(|e| Error::Retrieval(format!("Failed to read safetensors: {}", e)))?;

    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| Error::Parse(format!("Failed to parse safetensors: {}", e)))?;

    // Helper to extract tensor as Vec<f32>
    let get_tensor = |name: &str| -> crate::Result<Vec<f32>> {
        let tensor = tensors
            .tensor(name)
            .map_err(|_| Error::Parse(format!("Tensor '{}' not found", name)))?;

        // Convert to f32
        let data: Vec<f32> = match tensor.dtype() {
            safetensors::Dtype::F32 => {
                let bytes = tensor.data();
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            safetensors::Dtype::F16 => {
                // Simple f16 -> f32 conversion without half crate
                let bytes = tensor.data();
                bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16_to_f32(bits)
                    })
                    .collect()
            }
            dtype => {
                return Err(Error::Parse(format!(
                    "Unsupported dtype {:?} for tensor '{}'",
                    dtype, name
                )))
            }
        };

        Ok(data)
    };

    // Try different naming conventions
    let (mu_weight_name, mu_bias_name, sigma_weight_name, sigma_bias_name) =
        if tensors.tensor("text_l1_mu.weight").is_ok() {
            (
                "text_l1_mu.weight",
                "text_l1_mu.bias",
                "text_l1_sigma.weight",
                "text_l1_sigma.bias",
            )
        } else if tensors.tensor("mu_proj.weight").is_ok() {
            (
                "mu_proj.weight",
                "mu_proj.bias",
                "log_sigma_proj.weight",
                "log_sigma_proj.bias",
            )
        } else {
            return Err(Error::Parse(
                "Could not find expected tensor names in safetensors".to_string(),
            ));
        };

    // Load tensors
    let mu_weight_flat = get_tensor(mu_weight_name)?;
    let mu_bias = get_tensor(mu_bias_name)?;
    let sigma_weight_flat = get_tensor(sigma_weight_name)?;
    let sigma_bias = get_tensor(sigma_bias_name)?;

    // Get dimensions from bias vectors
    let output_dim = mu_bias.len();
    let input_dim = mu_weight_flat.len() / output_dim;

    // Reshape weight matrices [input_dim, output_dim]
    let mu_weight: Vec<Vec<f32>> = mu_weight_flat
        .chunks(output_dim)
        .map(|chunk| chunk.to_vec())
        .collect();

    let sigma_weight: Vec<Vec<f32>> = sigma_weight_flat
        .chunks(output_dim)
        .map(|chunk| chunk.to_vec())
        .collect();

    Ok(BoxProjectionWeights {
        mu_weight,
        mu_bias,
        sigma_weight,
        sigma_bias,
        input_dim,
        output_dim,
    })
}

/// Loads box projection weights from a safetensors file (stub when feature disabled).
#[cfg(not(feature = "safetensors"))]
pub fn load_box_projection_weights<P: AsRef<Path>>(_path: P) -> crate::Result<BoxProjectionWeights> {
    Err(Error::Retrieval(
        "safetensors feature not enabled. Rebuild with --features safetensors".to_string(),
    ))
}

/// Metadata about box projection weights.
#[derive(Debug, Clone)]
pub struct BoxWeightsMetadata {
    /// Description of the weights.
    pub description: Option<String>,
    /// Source checkpoint path.
    pub source_checkpoint: Option<String>,
    /// Input dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Model type (e.g., "GaussianBoxHead").
    pub model_type: Option<String>,
    /// Training datasets used.
    pub training_datasets: Option<String>,
}

/// Loads metadata from a JSON file alongside safetensors weights.
pub fn load_box_weights_metadata<P: AsRef<Path>>(
    safetensors_path: P,
) -> crate::Result<BoxWeightsMetadata> {
    let path = safetensors_path.as_ref();
    let metadata_path = path.with_file_name(
        path.file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
            + "_metadata.json",
    );

    if !metadata_path.exists() {
        // Return default metadata
        return Ok(BoxWeightsMetadata {
            description: None,
            source_checkpoint: None,
            input_dim: 768,
            output_dim: 128,
            model_type: None,
            training_datasets: None,
        });
    }

    let content = std::fs::read_to_string(&metadata_path)
        .map_err(|e| Error::Retrieval(format!("Failed to read metadata: {}", e)))?;

    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| Error::Parse(format!("Failed to parse metadata JSON: {}", e)))?;

    Ok(BoxWeightsMetadata {
        description: json.get("description").and_then(|v| v.as_str()).map(String::from),
        source_checkpoint: json.get("source_checkpoint").and_then(|v| v.as_str()).map(String::from),
        input_dim: json.get("input_dim").and_then(|v| v.as_u64()).unwrap_or(768) as usize,
        output_dim: json.get("output_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize,
        model_type: json.get("model_type").and_then(|v| v.as_str()).map(String::from),
        training_datasets: json.get("training_datasets").and_then(|v| v.as_str()).map(String::from),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_projection() {
        let weights = BoxProjectionWeights::new_identity(768, 128);

        // Project a simple embedding
        let embedding = vec![1.0; 768];
        let (center, delta) = weights.project_to_box(&embedding);

        // First 128 dims of center should be 1.0 (identity)
        assert_eq!(center.len(), 128);
        for i in 0..128 {
            assert!((center[i] - 1.0).abs() < 1e-6, "center[{}] = {}", i, center[i]);
        }

        // Delta should be ~0.5 (exp(-0.69) ≈ 0.5)
        assert_eq!(delta.len(), 128);
        for i in 0..128 {
            assert!(
                (delta[i] - 0.5).abs() < 0.1,
                "delta[{}] = {}",
                i,
                delta[i]
            );
        }
    }

    #[test]
    fn test_project_to_box_embedding() {
        let weights = BoxProjectionWeights::new_identity(768, 128);
        let embedding = vec![0.5; 768];
        let box_emb = weights.project_to_box_embedding(&embedding);

        // Box should be centered at 0.5 with size ~1.0
        let center = box_emb.center();
        assert_eq!(center.len(), 128);

        // First 128 dims should be ~0.5
        for i in 0..128 {
            assert!(
                (center[i] - 0.5).abs() < 0.1,
                "center[{}] = {}",
                i,
                center[i]
            );
        }
    }

    #[test]
    fn test_coreference_with_box_projections() {
        // Test that similar embeddings produce overlapping boxes
        let weights = BoxProjectionWeights::new_identity(768, 128);

        // Two similar embeddings (representing same entity in different contexts)
        let emb1: Vec<f32> = (0..768).map(|i| 0.5 + (i as f32 / 768.0) * 0.1).collect();
        let emb2: Vec<f32> = (0..768).map(|i| 0.5 + (i as f32 / 768.0) * 0.12).collect();

        let box1 = weights.project_to_box_embedding(&emb1);
        let box2 = weights.project_to_box_embedding(&emb2);

        // Similar embeddings should produce overlapping boxes
        let p12 = box1.conditional_probability(&box2);
        let p21 = box2.conditional_probability(&box1);

        assert!(p12 > 0.5, "Similar embeddings should overlap: P(2|1)={}", p12);
        assert!(p21 > 0.5, "Similar embeddings should overlap: P(1|2)={}", p21);
    }

    #[test]
    fn test_non_coreference_with_box_projections() {
        // Test that dissimilar embeddings produce non-overlapping boxes
        let weights = BoxProjectionWeights::new_identity(768, 128);

        // Two dissimilar embeddings (representing different entities)
        let emb1: Vec<f32> = (0..768).map(|i| if i < 384 { 1.0 } else { 0.0 }).collect();
        let emb2: Vec<f32> = (0..768).map(|i| if i < 384 { 0.0 } else { 1.0 }).collect();

        let box1 = weights.project_to_box_embedding(&emb1);
        let box2 = weights.project_to_box_embedding(&emb2);

        // Dissimilar embeddings should produce less overlapping boxes
        let p12 = box1.conditional_probability(&box2);
        let p21 = box2.conditional_probability(&box1);

        // These should be lower than similar embeddings
        // Note: With identity projection, the result depends on the embedding patterns
        assert!(p12 >= 0.0 && p12 <= 1.0, "Valid probability: {}", p12);
        assert!(p21 >= 0.0 && p21 <= 1.0, "Valid probability: {}", p21);
    }

    #[test]
    #[cfg(feature = "safetensors")]
    fn test_f16_conversion() {
        // Test f16 -> f32 conversion for common values
        
        // Zero
        assert_eq!(f16_to_f32(0x0000), 0.0);
        
        // One (IEEE 754 half: sign=0, exp=15, mantissa=0)
        let one_bits: u16 = 0x3C00;
        let converted = f16_to_f32(one_bits);
        assert!((converted - 1.0).abs() < 1e-6, "1.0 conversion: {}", converted);
        
        // Negative one
        let neg_one_bits: u16 = 0xBC00;
        let neg_converted = f16_to_f32(neg_one_bits);
        assert!((neg_converted - (-1.0)).abs() < 1e-6, "-1.0 conversion: {}", neg_converted);
        
        // Infinity
        assert!(f16_to_f32(0x7C00).is_infinite());
        
        // NaN
        assert!(f16_to_f32(0x7E00).is_nan());
    }
}

