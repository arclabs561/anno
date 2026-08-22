//! Span enumeration, mention scoring, and antecedent scoring for f-coref.
//!
//! Implements the scorer heads from the f-coref architecture (Otmazgin et al., AACL 2022)
//! as pure Rust using ndarray, loading pre-trained weights from safetensors.
//!
//! The scoring pipeline:
//! 1. Generate span candidates (i, j) where j - i < max_span_length
//! 2. Score each span using start/end MLPs + bilinear interaction
//! 3. Keep top-k mentions (k = seq_len * top_lambda)
//! 4. Score antecedent pairs using four bilinear classifiers (s2s, e2e, s2e, e2s)
//! 5. For each mention, pick the best antecedent (or null = self)

use ndarray::{s, Array1, Array2};

use crate::{Error, Result};

/// Output of `score_mentions`: top-k mention spans with their coref representations.
#[derive(Debug)]
pub(crate) struct MentionScoringResult {
    pub top_k_starts: Vec<usize>,
    pub top_k_ends: Vec<usize>,
    pub top_k_logits: Vec<f32>,
    pub start_coref_reps: Array2<f32>,
    pub end_coref_reps: Array2<f32>,
}

/// Weights for a single FullyConnectedLayer (Linear + LayerNorm, no dropout at inference).
#[derive(Debug)]
pub(crate) struct FcLayerWeights {
    pub linear_weight: Array2<f32>, // [out, in]
    pub linear_bias: Array1<f32>,   // [out]
    pub norm_weight: Array1<f32>,   // [out]
    pub norm_bias: Array1<f32>,     // [out]
}

impl FcLayerWeights {
    /// Apply: LayerNorm(GELU(Wx + b)).
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // Linear: [batch, in] @ [in, out] + bias
        let linear_out = input.dot(&self.linear_weight.t()) + &self.linear_bias;

        // GELU activation
        let gelu_out = gelu(&linear_out);

        // LayerNorm
        layer_norm(&gelu_out, &self.norm_weight, &self.norm_bias, 1e-5)
    }
}

/// Weights for a classifier (Linear layer: Wx + b or just Wx).
#[derive(Debug)]
pub(crate) struct ClassifierWeights {
    pub weight: Array2<f32>, // [out, in]
    pub bias: Option<Array1<f32>>,
}

impl ClassifierWeights {
    /// Apply linear transform.
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let out = input.dot(&self.weight.t());
        if let Some(ref bias) = self.bias {
            out + bias
        } else {
            out
        }
    }
}

/// All scorer head weights loaded from safetensors.
pub(crate) struct ScorerWeights {
    // Mention detection
    pub start_mention_mlp: FcLayerWeights,
    pub end_mention_mlp: FcLayerWeights,
    pub mention_start_classifier: ClassifierWeights,
    pub mention_end_classifier: ClassifierWeights,
    pub mention_s2e_classifier: ClassifierWeights,

    // Coreference
    pub start_coref_mlp: FcLayerWeights,
    pub end_coref_mlp: FcLayerWeights,
    pub antecedent_s2s_classifier: ClassifierWeights,
    pub antecedent_e2e_classifier: ClassifierWeights,
    pub antecedent_s2e_classifier: ClassifierWeights,
    pub antecedent_e2s_classifier: ClassifierWeights,
}

impl ScorerWeights {
    /// Load from a safetensors file.
    pub fn from_safetensors(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| Error::Retrieval(format!("Failed to read scorer weights: {}", e)))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| Error::Parse(format!("Failed to parse safetensors: {}", e)))?;

        // Helper to load an ndarray from safetensors
        let load_2d = |name: &str| -> Result<Array2<f32>> {
            let view = tensors
                .tensor(name)
                .map_err(|e| Error::Parse(format!("Missing tensor '{}': {}", name, e)))?;
            let shape = view.shape();
            if shape.len() != 2 {
                return Err(Error::Parse(format!(
                    "Expected 2D tensor for '{}', got {:?}",
                    name, shape
                )));
            }
            let data: Vec<f32> = view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Array2::from_shape_vec((shape[0], shape[1]), data)
                .map_err(|e| Error::Parse(format!("Shape mismatch for '{}': {}", name, e)))
        };

        let load_1d = |name: &str| -> Result<Array1<f32>> {
            let view = tensors
                .tensor(name)
                .map_err(|e| Error::Parse(format!("Missing tensor '{}': {}", name, e)))?;
            let shape = view.shape();
            if shape.len() != 1 {
                return Err(Error::Parse(format!(
                    "Expected 1D tensor for '{}', got {:?}",
                    name, shape
                )));
            }
            let data: Vec<f32> = view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(Array1::from_vec(data))
        };

        let load_fc = |prefix: &str| -> Result<FcLayerWeights> {
            Ok(FcLayerWeights {
                linear_weight: load_2d(&format!("{}.dense.weight", prefix))?,
                linear_bias: load_1d(&format!("{}.dense.bias", prefix))?,
                norm_weight: load_1d(&format!("{}.layer_norm.weight", prefix))?,
                norm_bias: load_1d(&format!("{}.layer_norm.bias", prefix))?,
            })
        };

        let load_classifier = |prefix: &str| -> Result<ClassifierWeights> {
            let weight = load_2d(&format!("{}.weight", prefix))?;
            let bias = load_1d(&format!("{}.bias", prefix)).ok();
            Ok(ClassifierWeights { weight, bias })
        };

        Ok(Self {
            start_mention_mlp: load_fc("start_mention_mlp")?,
            end_mention_mlp: load_fc("end_mention_mlp")?,
            mention_start_classifier: load_classifier("mention_start_classifier")?,
            mention_end_classifier: load_classifier("mention_end_classifier")?,
            mention_s2e_classifier: load_classifier("mention_s2e_classifier")?,
            start_coref_mlp: load_fc("start_coref_mlp")?,
            end_coref_mlp: load_fc("end_coref_mlp")?,
            antecedent_s2s_classifier: load_classifier("antecedent_s2s_classifier")?,
            antecedent_e2e_classifier: load_classifier("antecedent_e2e_classifier")?,
            antecedent_s2e_classifier: load_classifier("antecedent_s2e_classifier")?,
            antecedent_e2s_classifier: load_classifier("antecedent_e2s_classifier")?,
        })
    }
}

/// Compute mention logits from hidden states.
///
/// Returns mention logits as a flattened score vector with (start, end) indices,
/// plus the top-k selected mention indices.
///
/// # Arguments
///
/// * `hidden` - Encoder hidden states [seq_len, hidden_size]
/// * `weights` - Scorer weights
/// * `max_span_length` - Maximum span width
/// * `top_lambda` - Fraction of sequence to keep as mentions
///
/// # Returns
///
/// A [`MentionScoringResult`] with top-k mention spans and coref representations.
pub(crate) fn score_mentions(
    hidden: &Array2<f32>,
    weights: &ScorerWeights,
    max_span_length: usize,
    top_lambda: f32,
) -> MentionScoringResult {
    let seq_len = hidden.nrows();

    // Compute mention representations
    let start_mention_reps = weights.start_mention_mlp.forward(hidden);
    let end_mention_reps = weights.end_mention_mlp.forward(hidden);

    // Compute start/end logits: [seq_len, 1] -> [seq_len]
    let start_logits = weights
        .mention_start_classifier
        .forward(&start_mention_reps)
        .column(0)
        .to_owned();
    let end_logits = weights
        .mention_end_classifier
        .forward(&end_mention_reps)
        .column(0)
        .to_owned();

    // Compute bilinear s2e interaction: [seq_len, ffnn] @ [ffnn, seq_len] -> [seq_len, seq_len]
    let s2e_transformed = weights.mention_s2e_classifier.forward(&start_mention_reps);
    let joint_logits = s2e_transformed.dot(&end_mention_reps.t());

    // Combine: mention_logits[i, j] = start[i] + end[j] + joint[i, j]
    // Then mask to valid spans (i <= j < i + max_span_length)
    let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..seq_len {
        let j_max = (i + max_span_length).min(seq_len);
        for j in i..j_max {
            let score = start_logits[i] + end_logits[j] + joint_logits[[i, j]];
            candidates.push((i, j, score));
        }
    }

    // Top-k selection
    let k = ((seq_len as f32 * top_lambda).ceil() as usize).max(1);
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(k);
    // Re-sort by position for deterministic ordering
    candidates.sort_by_key(|&(start, end, _)| (start, end));

    let top_k_starts: Vec<usize> = candidates.iter().map(|c| c.0).collect();
    let top_k_ends: Vec<usize> = candidates.iter().map(|c| c.1).collect();
    let top_k_logits: Vec<f32> = candidates.iter().map(|c| c.2).collect();

    // Compute coref representations (needed for antecedent scoring)
    let start_coref_reps = weights.start_coref_mlp.forward(hidden);
    let end_coref_reps = weights.end_coref_mlp.forward(hidden);

    MentionScoringResult {
        top_k_starts,
        top_k_ends,
        top_k_logits,
        start_coref_reps,
        end_coref_reps,
    }
}

/// Score antecedent pairs and return best antecedent for each mention.
///
/// For each mention i, computes the coreference score with all preceding mentions j < i,
/// plus a null antecedent score (the mention's own score). Returns the best antecedent
/// index per mention (or self-index for null).
///
/// # Arguments
///
/// * `top_k_starts` - Start token indices of top-k mentions
/// * `top_k_ends` - End token indices of top-k mentions
/// * `top_k_logits` - Mention scores for top-k
/// * `start_coref_reps` - Full coref start representations [seq_len, ffnn]
/// * `end_coref_reps` - Full coref end representations [seq_len, ffnn]
/// * `weights` - Scorer weights
///
/// # Returns
///
/// For each of the k mentions, the index of its best antecedent (0..k-1), or self for null.
pub(crate) fn score_antecedents(
    top_k_starts: &[usize],
    top_k_ends: &[usize],
    top_k_logits: &[f32],
    start_coref_reps: &Array2<f32>,
    end_coref_reps: &Array2<f32>,
    weights: &ScorerWeights,
) -> Vec<usize> {
    let k = top_k_starts.len();
    if k == 0 {
        return vec![];
    }

    let ffnn = start_coref_reps.ncols();

    // Gather coref representations for top-k mentions
    let mut top_start_reps = Array2::<f32>::zeros((k, ffnn));
    let mut top_end_reps = Array2::<f32>::zeros((k, ffnn));
    for (idx, (&si, &ei)) in top_k_starts.iter().zip(top_k_ends.iter()).enumerate() {
        top_start_reps
            .slice_mut(s![idx, ..])
            .assign(&start_coref_reps.slice(s![si, ..]));
        top_end_reps
            .slice_mut(s![idx, ..])
            .assign(&end_coref_reps.slice(s![ei, ..]));
    }

    // Compute four bilinear terms: [k, ffnn] @ [ffnn, k] -> [k, k]
    let s2s = weights
        .antecedent_s2s_classifier
        .forward(&top_start_reps)
        .dot(&top_start_reps.t());
    let e2e = weights
        .antecedent_e2e_classifier
        .forward(&top_end_reps)
        .dot(&top_end_reps.t());
    let s2e = weights
        .antecedent_s2e_classifier
        .forward(&top_start_reps)
        .dot(&top_end_reps.t());
    let e2s = weights
        .antecedent_e2s_classifier
        .forward(&top_end_reps)
        .dot(&top_start_reps.t());

    // Sum all four terms
    let coref_logits = &s2s + &e2e + &s2e + &e2s;

    // Add mention pair scores: score(i) + score(j) + coref(i, j)
    let mut antecedents = Vec::with_capacity(k);
    for i in 0..k {
        let mut best_score = 0.0_f32; // null antecedent score
        let mut best_ante = i; // self = null

        for j in 0..i {
            let score = top_k_logits[i] + top_k_logits[j] + coref_logits[[i, j]];
            if score > best_score {
                best_score = score;
                best_ante = j;
            }
        }

        antecedents.push(best_ante);
    }

    antecedents
}

// =============================================================================
// Activation and normalization functions
// =============================================================================

/// GELU activation (approximation used by PyTorch).
fn gelu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| {
        0.5 * v
            * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
    })
}

/// Layer normalization over the last axis.
fn layer_norm(x: &Array2<f32>, weight: &Array1<f32>, bias: &Array1<f32>, eps: f32) -> Array2<f32> {
    let n = x.ncols() as f32;
    let mut out = Array2::zeros(x.raw_dim());

    for (i, row) in x.rows().into_iter().enumerate() {
        let mean = row.sum() / n;
        let var = row.mapv(|v| (v - mean).powi(2)).sum() / n;
        let std = (var + eps).sqrt();

        for (j, &v) in row.iter().enumerate() {
            out[[i, j]] = (v - mean) / std * weight[j] + bias[j];
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gelu_zero() {
        let x = Array2::zeros((1, 3));
        let out = gelu(&x);
        for &v in out.iter() {
            assert!((v - 0.0).abs() < 1e-6, "GELU(0) should be 0");
        }
    }

    #[test]
    fn test_gelu_positive() {
        let x = array![[1.0, 2.0, 3.0]];
        let out = gelu(&x);
        // GELU is approximately identity for large positive values
        assert!(out[[0, 0]] > 0.8);
        assert!(out[[0, 1]] > 1.9);
        assert!(out[[0, 2]] > 2.9);
    }

    #[test]
    fn test_layer_norm_constant_input() {
        let x = array![[5.0, 5.0, 5.0]];
        let w = Array1::ones(3);
        let b = Array1::zeros(3);
        let out = layer_norm(&x, &w, &b, 1e-5);
        // Constant input -> zero mean, zero variance -> all zeros (normalized)
        for &v in out.iter() {
            assert!(v.abs() < 1e-3, "LayerNorm of constant should be ~0");
        }
    }

    #[test]
    fn test_fc_layer_shape() {
        let fc = FcLayerWeights {
            linear_weight: Array2::zeros((4, 3)),
            linear_bias: Array1::zeros(4),
            norm_weight: Array1::ones(4),
            norm_bias: Array1::zeros(4),
        };
        let input = Array2::zeros((2, 3));
        let output = fc.forward(&input);
        assert_eq!(output.shape(), &[2, 4]);
    }

    #[test]
    fn test_classifier_shape() {
        let cls = ClassifierWeights {
            weight: Array2::zeros((1, 4)),
            bias: Some(Array1::zeros(1)),
        };
        let input = Array2::zeros((5, 4));
        let output = cls.forward(&input);
        assert_eq!(output.shape(), &[5, 1]);
    }

    #[test]
    fn test_score_antecedents_single_mention() {
        // Single mention: should point to self (null antecedent)
        let starts = vec![0];
        let ends = vec![1];
        let logits = vec![1.0];
        let start_reps = Array2::zeros((10, 4));
        let end_reps = Array2::zeros((10, 4));

        let weights = dummy_scorer_weights(4);
        let antes = score_antecedents(&starts, &ends, &logits, &start_reps, &end_reps, &weights);
        assert_eq!(antes.len(), 1);
        assert_eq!(antes[0], 0); // self = null
    }

    /// Create dummy scorer weights for testing.
    fn dummy_scorer_weights(ffnn: usize) -> ScorerWeights {
        let fc = |in_dim: usize, out_dim: usize| FcLayerWeights {
            linear_weight: Array2::zeros((out_dim, in_dim)),
            linear_bias: Array1::zeros(out_dim),
            norm_weight: Array1::ones(out_dim),
            norm_bias: Array1::zeros(out_dim),
        };
        let cls = |in_dim: usize, out_dim: usize| ClassifierWeights {
            weight: Array2::zeros((out_dim, in_dim)),
            bias: Some(Array1::zeros(out_dim)),
        };

        ScorerWeights {
            start_mention_mlp: fc(768, ffnn),
            end_mention_mlp: fc(768, ffnn),
            mention_start_classifier: cls(ffnn, 1),
            mention_end_classifier: cls(ffnn, 1),
            mention_s2e_classifier: cls(ffnn, ffnn),
            start_coref_mlp: fc(768, ffnn),
            end_coref_mlp: fc(768, ffnn),
            antecedent_s2s_classifier: cls(ffnn, ffnn),
            antecedent_e2e_classifier: cls(ffnn, ffnn),
            antecedent_s2e_classifier: cls(ffnn, ffnn),
            antecedent_e2s_classifier: cls(ffnn, ffnn),
        }
    }
}
