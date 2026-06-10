//! `span_rep` head — `SpanMarkerV0` ported to Candle (M5a).
//!
//! See `docs/dev-notes/gliner2-multi-v1-forward-pass.md` §2 and
//! `docs/dev-notes/gliner2-multi-v1-safetensors-keys.md` ("span_rep") for
//! the authoritative spec. In short: three 2-layer MLPs (`project_start`,
//! `project_end`, `out_project`) with ReLU between layers and a Dropout
//! that's a no-op at inference. Indices `0` and `3` in each PyTorch
//! `nn.Sequential` are the two `Linear`s; indices `1` and `2` are
//! ReLU + Dropout (no params).
//!
//! Build:
//!   `start_rep = linear_3(relu(linear_0(text_emb)))`     # [1, T, 768]
//!   `end_rep   = linear_3(relu(linear_0(text_emb)))`     # [1, T, 768]
//!   gather `start_at[i, j, :] = start_rep[i, span_idx[i, j, 0], :]`
//!   gather `end_at  [i, j, :] = end_rep  [i, span_idx[i, j, 1], :]`
//!   `cat = relu(concat(start_at, end_at, dim=-1))`       # [1, T*W, 1536]
//!   `out = linear_3(relu(linear_0(cat)))`                # [1, T*W, 768]
//!   reshape → [1, T, MAX_WIDTH, 768]

use candle_core::{IndexOp, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::backends::gliner2_fastino::pipeline::MAX_WIDTH;

/// SpanMarkerV0 — builds per-(start, end) span representations from
/// per-token hidden states.
pub struct SpanRep {
    /// project_start.0 : Linear 768 → 3072
    project_start_0: Linear,
    /// project_start.3 : Linear 3072 → 768
    project_start_3: Linear,
    /// project_end.0   : Linear 768 → 3072
    project_end_0: Linear,
    /// project_end.3   : Linear 3072 → 768
    project_end_3: Linear,
    /// out_project.0   : Linear 1536 → 3072
    out_project_0: Linear,
    /// out_project.3   : Linear 3072 → 768
    out_project_3: Linear,
}

impl SpanRep {
    /// Construct from a `VarBuilder` rooted at `span_rep.span_rep_layer`.
    ///
    /// Reads the 12 keys documented in
    /// `docs/dev-notes/gliner2-multi-v1-safetensors-keys.md`:
    ///   - `project_start.0.{weight,bias}`   (3072, 768) / (3072,)
    ///   - `project_start.3.{weight,bias}`   (768, 3072) / (768,)
    ///   - `project_end.0.{weight,bias}`     (3072, 768) / (3072,)
    ///   - `project_end.3.{weight,bias}`     (768, 3072) / (768,)
    ///   - `out_project.0.{weight,bias}`     (3072, 1536) / (3072,)
    ///   - `out_project.3.{weight,bias}`     (768, 3072) / (768,)
    pub fn from_var_builder(vb: &VarBuilder) -> candle_core::Result<Self> {
        let project_start_0 = linear(768, 3072, vb.pp("project_start.0"))?;
        let project_start_3 = linear(3072, 768, vb.pp("project_start.3"))?;
        let project_end_0 = linear(768, 3072, vb.pp("project_end.0"))?;
        let project_end_3 = linear(3072, 768, vb.pp("project_end.3"))?;
        let out_project_0 = linear(1536, 3072, vb.pp("out_project.0"))?;
        let out_project_3 = linear(3072, 768, vb.pp("out_project.3"))?;

        Ok(Self {
            project_start_0,
            project_start_3,
            project_end_0,
            project_end_3,
            out_project_0,
            out_project_3,
        })
    }

    /// Forward pass.
    ///
    /// * `text_emb` — `[1, T, 768]` per-word pooled hidden states.
    /// * `span_idx` — `[1, T*MAX_WIDTH, 2]` int64 (start, end) indices
    ///   into the T axis. Out-of-range pairs must be pre-clamped to a
    ///   safe (start, end) (the upstream `_compute_span_rep_core` uses
    ///   `(0, 0)` for invalid spans).
    ///
    /// Returns `[1, T, MAX_WIDTH, 768]`.
    pub fn forward(&self, text_emb: &Tensor, span_idx: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _h) = text_emb.dims3()?;
        debug_assert_eq!(b, 1, "SpanRep currently assumes batch=1");

        // Per-token start / end projections: 768 → 3072 → 768 with ReLU
        // between. Dropout (Sequential index 2) is identity at inference.
        let start_rep = self
            .project_start_3
            .forward(&self.project_start_0.forward(text_emb)?.relu()?)?;
        let end_rep = self
            .project_end_3
            .forward(&self.project_end_0.forward(text_emb)?.relu()?)?;

        // Gather start / end token reps at the indices in `span_idx`.
        // Candle's `index_select` wants 1-D indices, so squeeze the batch
        // dim, gather along dim=0, then unsqueeze back. This matches the
        // PyTorch `extract_elements`'s `gather` (which broadcasts a
        // (B, N, 1)→(B, N, H) index along the hidden dim — equivalent).
        //
        // `span_idx`: [1, T*W, 2] i64 → take .., 0 / .., 1 → [1, T*W] →
        // squeeze → [T*W].
        let start_idx = span_idx.i((0, .., 0))?.contiguous()?; // [T*W]
        let end_idx = span_idx.i((0, .., 1))?.contiguous()?; // [T*W]

        let start_rep_2d = start_rep.squeeze(0)?; // [T, 768]
        let end_rep_2d = end_rep.squeeze(0)?; // [T, 768]

        let start_at = start_rep_2d.index_select(&start_idx, 0)?; // [T*W, 768]
        let end_at = end_rep_2d.index_select(&end_idx, 0)?; // [T*W, 768]

        // cat = relu(concat([start, end], dim=-1)) ; [T*W, 1536]
        let cat = Tensor::cat(&[&start_at, &end_at], 1)?.relu()?;

        // out_project: 1536 → 3072 → 768 with ReLU between.
        let out_2d = self
            .out_project_3
            .forward(&self.out_project_0.forward(&cat)?.relu()?)?; // [T*W, 768]

        // Reshape to [1, T, MAX_WIDTH, 768].
        let out = out_2d.reshape((1, t, MAX_WIDTH, 768))?;
        Ok(out)
    }
}
