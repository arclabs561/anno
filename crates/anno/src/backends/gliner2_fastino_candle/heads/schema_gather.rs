//! `schema_gather` — non-parametric utility head.
//!
//! Performs `index_select` at the seq positions of the
//! `[P]` / `[E]` / `[L]` / `[C]` / `[R]` special tokens recorded by
//! the processor. See `docs/dev-notes/gliner2-multi-v1-forward-pass.md`
//! §2 step 2 for the spec.

use candle_core::{Result, Tensor};

/// Stateless schema-gather head. Holds no parameters.
pub struct SchemaGather;

/// Result of [`SchemaGather::forward`].
pub struct SchemaGatherOutput {
    /// `[1, H]` — the `[P]` token's hidden state (prompt context).
    pub pc_emb: Tensor,
    /// `[F, H]` — per-field / per-label embeddings (`[E]`/`[L]`/`[C]`/`[R]`).
    pub field_embs: Tensor,
}

impl SchemaGather {
    /// Take `hidden_states[0, schema_indices[0], :]` as `pc_emb` and
    /// `hidden_states[0, schema_indices[1..], :]` as `field_embs`.
    ///
    /// `schema_indices` includes the `[P]` index first, followed by
    /// all per-field indices — matches the order
    /// `processor::TaskMapping` stores them in (re-exported from the
    /// ONNX backend).
    pub fn forward(
        &self,
        hidden_states: &Tensor,  // [1, S, H]
        schema_indices: &Tensor, // [num_special]
    ) -> Result<SchemaGatherOutput> {
        let h = hidden_states.squeeze(0)?; // [S, H]
        let all = h.index_select(schema_indices, 0)?; // [num_special, H]

        // Row 0 is [P]; rows 1..num_special are field-level.
        let pc_emb = all.narrow(0, 0, 1)?; // [1, H]
        let n = all.dim(0)?;
        let hidden_dim = all.dim(1)?;
        let field_embs = if n > 1 {
            all.narrow(0, 1, n - 1)? // [F, H]
        } else {
            // Degenerate case: no fields.
            Tensor::zeros((0, hidden_dim), all.dtype(), all.device())?
        };

        Ok(SchemaGatherOutput { pc_emb, field_embs })
    }
}
