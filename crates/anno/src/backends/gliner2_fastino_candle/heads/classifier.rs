//! `classifier` head — 2-layer MLP that scores a single field's labels,
//! ported to Candle (M5b).
//!
//! See `docs/dev-notes/gliner2-multi-v1-safetensors-keys.md` ("classifier")
//! and `docs/dev-notes/gliner2-multi-v1-forward-pass.md` §6 for the spec.
//!
//! PyTorch layout (`nn.Sequential`):
//!
//! ```text
//!   0: Linear 768 → 1536
//!   1: ReLU                  (no params)
//!   2: Linear 1536 → 1
//! ```
//!
//! Forward pass:
//!
//! ```text
//!   h1     = relu(linear_0(field_embs))   # [num_labels, 1536]
//!   logits = linear_2(h1)                 # [num_labels, 1]
//!   out    = logits.squeeze(-1)           # [num_labels]
//! ```
//!
//! Returns raw logits — the caller decides whether to apply sigmoid
//! (multi-label) or softmax (single-label) downstream.

use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};

/// `classifier` — per-label scoring MLP.
pub struct Classifier {
    /// `classifier.0` : Linear 768 → 1536
    linear_0: Linear,
    /// `classifier.2` : Linear 1536 → 1
    linear_2: Linear,
}

impl Classifier {
    /// Construct from a `VarBuilder` rooted at `classifier`.
    ///
    /// Reads:
    ///   - `0.{weight,bias}`  (1536, 768) / (1536,)
    ///   - `2.{weight,bias}`  (1, 1536)   / (1,)
    pub fn from_var_builder(vb: &VarBuilder) -> candle_core::Result<Self> {
        let linear_0 = linear(768, 1536, vb.pp("0"))?;
        let linear_2 = linear(1536, 1, vb.pp("2"))?;
        Ok(Self { linear_0, linear_2 })
    }

    /// Forward pass.
    ///
    /// * `field_embs` — `[num_labels, 768]` per-label embeddings.
    ///
    /// Returns `[num_labels]` raw logits (no sigmoid / softmax applied —
    /// caller chooses single- vs multi-label normalisation).
    pub fn forward(&self, field_embs: &Tensor) -> candle_core::Result<Tensor> {
        // h1 = relu(linear_0(field_embs))  → [num_labels, 1536]
        let h1 = self.linear_0.forward(field_embs)?.relu()?;
        // logits = linear_2(h1)            → [num_labels, 1]
        let logits = self.linear_2.forward(&h1)?;
        // squeeze last dim → [num_labels]
        logits.squeeze(1)
    }
}
