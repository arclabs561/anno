//! `count_pred` head — 2-layer MLP over the pooled prompt embedding,
//! ported to Candle (M5b).
//!
//! See `docs/dev-notes/gliner2-multi-v1-safetensors-keys.md` ("count_pred")
//! and `docs/dev-notes/gliner2-multi-v1-forward-pass.md` §4 for the spec.
//!
//! PyTorch layout (`nn.Sequential`):
//!
//! ```text
//!   0: Linear 768 → 1536
//!   1: ReLU                  (no params)
//!   2: Linear 1536 → 20
//! ```
//!
//! Forward pass:
//!
//! ```text
//!   h1     = relu(linear_0(p_emb))    # [1, 1536]
//!   logits = linear_2(h1)             # [1, 20]
//!   pred   = logits.argmax(dim=1)     # scalar usize, clamped to [0, 19]
//! ```
//!
//! The return type is `usize` rather than `Tensor` because the value
//! drives host-side control flow (the GRU loop length in
//! [`super::count_lstm::CountLstmFixed`]).

use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Maximum count class index. Output dim is `MAX_COUNT_CLASSES = 20`,
/// so valid argmax results fall in `[0, 19]`.
const MAX_COUNT_CLASSES: usize = 20;

/// `count_pred` — 2-layer MLP that predicts a count class given the
/// pooled prompt embedding.
pub struct CountPred {
    /// `count_pred.0` : Linear 768 → 1536
    linear_0: Linear,
    /// `count_pred.2` : Linear 1536 → 20
    linear_2: Linear,
}

impl CountPred {
    /// Construct from a `VarBuilder` rooted at `count_pred`.
    ///
    /// Reads:
    ///   - `0.{weight,bias}`  (1536, 768) / (1536,)
    ///   - `2.{weight,bias}`  (20, 1536)  / (20,)
    pub fn from_var_builder(vb: &VarBuilder) -> candle_core::Result<Self> {
        let linear_0 = linear(768, 1536, vb.pp("0"))?;
        let linear_2 = linear(1536, MAX_COUNT_CLASSES, vb.pp("2"))?;
        Ok(Self { linear_0, linear_2 })
    }

    /// Forward pass.
    ///
    /// * `p_emb` — pooled prompt embedding of shape `[1, 768]` (or
    ///   `[768]`; reshaped internally).
    ///
    /// Returns the predicted count as a host-side `usize`, clamped to
    /// `[0, 19]` (the output of `argmax` over a 20-class logit vector).
    pub fn forward(&self, p_emb: &Tensor) -> candle_core::Result<usize> {
        // Normalise to [1, 768]: accept either [768] or [1, 768].
        let p_emb_2d = match p_emb.rank() {
            1 => p_emb.reshape((1, 768))?,
            2 => p_emb.clone(),
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "count_pred::forward: expected p_emb rank 1 or 2, got {other}"
                )));
            }
        };

        // h1 = relu(linear_0(p_emb))  → [1, 1536]
        let h1 = self.linear_0.forward(&p_emb_2d)?.relu()?;
        // logits = linear_2(h1)       → [1, 20]
        let logits = self.linear_2.forward(&h1)?;

        // argmax along the class axis. Reduces dim=1 → shape [1] → scalar.
        let argmax = logits.argmax(1)?; // [1], dtype u32
        let argmax_scalar = argmax.reshape(())?.to_scalar::<u32>()? as usize;

        Ok(argmax_scalar.min(MAX_COUNT_CLASSES - 1))
    }
}
