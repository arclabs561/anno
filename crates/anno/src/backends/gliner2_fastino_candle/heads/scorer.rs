//! `scorer` — non-parametric utility head.
//!
//! Implements the einsum-with-sigmoid step described in
//! `docs/dev-notes/gliner2-multi-v1-forward-pass.md` §"the scorer":
//!
//! ```text
//!   scores[b, p, l, k] = sigmoid(Σ_d span_rep[l, k, d] * struct_proj[b, p, d])
//! ```
//!
//! Candle has no `einsum`, but the contraction decomposes cleanly into
//! a single matmul + reshape + sigmoid.

use candle_core::{Result, Tensor};

/// Stateless scorer head. Holds no parameters.
pub struct Scorer;

impl Scorer {
    /// Compute `[count, F, T, W]` sigmoid scores from per-span and
    /// per-(instance, field) embeddings.
    ///
    /// * `span_rep`: `[T, W, H]` (per-sample slice of `[1, T, W, H]`).
    /// * `struct_proj`: `[count, F, H]`.
    pub fn forward(&self, span_rep: &Tensor, struct_proj: &Tensor) -> Result<Tensor> {
        let (t, w, h) = span_rep.dims3()?;
        let (count, f, h2) = struct_proj.dims3()?;
        if h != h2 {
            return Err(candle_core::Error::Msg(format!(
                "scorer: hidden mismatch {h} vs {h2}"
            )));
        }

        // Flatten and matmul:
        //   span_flat:   [T*W, H]
        //   struct_flat: [count*F, H]
        //   scores_flat = struct_flat @ span_flat^T → [count*F, T*W]
        let span_flat = span_rep.reshape(((), h))?.contiguous()?; // [T*W, H]
        let struct_flat = struct_proj.reshape(((), h))?.contiguous()?; // [count*F, H]
        let scores_flat = struct_flat.matmul(&span_flat.transpose(0, 1)?.contiguous()?)?;
        let scores = scores_flat.reshape((count, f, t, w))?;

        candle_nn::ops::sigmoid(&scores)
    }
}
