use super::*;

/// Span representation layer that computes span embeddings from token-level hidden states.
pub struct SpanRepLayer {
    /// MLP for projecting start positions (Linear -> ReLU -> Dropout -> Linear)
    project_start_0: Linear,
    project_start_3: Linear,
    /// MLP for projecting end positions
    project_end_0: Linear,
    project_end_3: Linear,
    /// Final projection layer
    out_project_0: Linear,
    out_project_3: Linear,
    hidden_size: usize,
}

#[cfg(feature = "candle")]
impl SpanRepLayer {
    /// Create a new span representation layer from GLiNER weights.
    ///
    /// GLiNER uses the SpanMarker architecture with:
    /// - project_start: Linear(D, 4D) -> ReLU -> Dropout -> Linear(4D, D)
    /// - project_end: Linear(D, 4D) -> ReLU -> Dropout -> Linear(4D, D)
    /// - out_project: Linear(2D, 4D) -> ReLU -> Dropout -> Linear(4D, D)
    pub fn new(hidden_size: usize, _max_width: usize, vb: VarBuilder) -> Result<Self> {
        // Load project_start MLP (layers 0 and 3, indices match PyTorch Sequential)
        // Hidden multiplier is 4x for these models
        let project_start_0 = linear(hidden_size, hidden_size * 4, vb.pp("project_start").pp("0"))
            .map_err(|e| Error::Retrieval(format!("SpanRepLayer project_start.0: {}", e)))?;
        let project_start_3 = linear(hidden_size * 4, hidden_size, vb.pp("project_start").pp("3"))
            .map_err(|e| Error::Retrieval(format!("SpanRepLayer project_start.3: {}", e)))?;

        // Load project_end MLP
        let project_end_0 = linear(hidden_size, hidden_size * 4, vb.pp("project_end").pp("0"))
            .map_err(|e| Error::Retrieval(format!("SpanRepLayer project_end.0: {}", e)))?;
        let project_end_3 = linear(hidden_size * 4, hidden_size, vb.pp("project_end").pp("3"))
            .map_err(|e| Error::Retrieval(format!("SpanRepLayer project_end.3: {}", e)))?;

        // Load out_project MLP (input is 2*hidden_size = concatenated start+end)
        let out_project_0 = linear(
            hidden_size * 2,
            hidden_size * 4,
            vb.pp("out_project").pp("0"),
        )
        .map_err(|e| Error::Retrieval(format!("SpanRepLayer out_project.0: {}", e)))?;
        let out_project_3 = linear(hidden_size * 4, hidden_size, vb.pp("out_project").pp("3"))
            .map_err(|e| Error::Retrieval(format!("SpanRepLayer out_project.3: {}", e)))?;

        Ok(Self {
            project_start_0,
            project_start_3,
            project_end_0,
            project_end_3,
            out_project_0,
            out_project_3,
            hidden_size,
        })
    }

    /// Compute span embeddings from token embeddings using SpanMarker approach.
    ///
    /// # Arguments
    /// * `token_embeddings` - [batch, seq_len, hidden]
    /// * `span_indices` - [batch, num_spans, 2] (start, end)
    ///
    /// # Returns
    /// [batch, num_spans, hidden]
    pub fn forward(&self, token_embeddings: &Tensor, span_indices: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden) = token_embeddings
            .dims3()
            .map_err(|e| Error::Parse(format!("token_embeddings dims: {}", e)))?;
        let (_, _num_spans, _) = span_indices
            .dims3()
            .map_err(|e| Error::Parse(format!("span_indices dims: {}", e)))?;

        // Project start and end representations for all tokens first
        // project_start: Linear -> ReLU (at layer 2, which is dropout in PyTorch) -> Linear
        let start_rep = self.project_start_0.forward(token_embeddings)?;
        let start_rep = start_rep.relu()?;
        let start_rep = self.project_start_3.forward(&start_rep)?;

        let end_rep = self.project_end_0.forward(token_embeddings)?;
        let end_rep = end_rep.relu()?;
        let end_rep = self.project_end_3.forward(&end_rep)?;

        // Extract start and end indices
        let start_idx = span_indices.i((.., .., 0))?.to_dtype(DType::U32)?;
        let end_idx = span_indices.i((.., .., 1))?.to_dtype(DType::U32)?;

        let mut span_embs = Vec::new();

        for b in 0..batch_size {
            let batch_start_rep = start_rep.i(b)?;
            let batch_end_rep = end_rep.i(b)?;
            let batch_starts = start_idx.i(b)?;
            let batch_ends = end_idx.i(b)?;

            // Clamp indices to valid range
            let max_idx = (seq_len - 1) as u32;
            let batch_starts = batch_starts.clamp(0f64, max_idx as f64)?;
            let batch_ends = batch_ends.clamp(0f64, max_idx as f64)?;

            // Extract start and end representations for each span
            let start_span_rep = batch_start_rep
                .index_select(&batch_starts.to_dtype(DType::U32)?, 0)
                .map_err(|e| Error::Parse(format!("start index_select: {}", e)))?;
            let end_span_rep = batch_end_rep
                .index_select(&batch_ends.to_dtype(DType::U32)?, 0)
                .map_err(|e| Error::Parse(format!("end index_select: {}", e)))?;

            // Concatenate and apply ReLU
            let cat = Tensor::cat(&[&start_span_rep, &end_span_rep], D::Minus1)?;
            let cat = cat.relu()?;

            // Apply output projection: Linear -> ReLU -> Linear
            let out = self.out_project_0.forward(&cat)?;
            let out = out.relu()?;
            let out = self.out_project_3.forward(&out)?;

            span_embs.push(out);
        }

        Tensor::stack(&span_embs, 0).map_err(|e| Error::Parse(format!("stack span_embs: {}", e)))
    }
}

// =============================================================================
// Label Encoder (prompt_rep_layer in GLiNER)
// =============================================================================

/// Projects label embeddings to matching space.
/// Maps to GLiNER's prompt_rep_layer MLP.
#[cfg(feature = "candle")]
pub struct LabelEncoder {
    linear_0: Linear,
    linear_3: Linear,
}

#[cfg(feature = "candle")]
impl LabelEncoder {
    /// Create a new label encoder from GLiNER prompt_rep_layer weights.
    ///
    /// GLiNER structure: Linear(D, 4D) -> ReLU -> Dropout -> Linear(4D, D)
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear_0 = linear(hidden_size, hidden_size * 4, vb.pp("0"))
            .map_err(|e| Error::Retrieval(format!("LabelEncoder.0: {}", e)))?;
        let linear_3 = linear(hidden_size * 4, hidden_size, vb.pp("3"))
            .map_err(|e| Error::Retrieval(format!("LabelEncoder.3: {}", e)))?;

        Ok(Self { linear_0, linear_3 })
    }

    /// Project label embeddings to matching space.
    pub fn forward(&self, label_embeddings: &Tensor) -> Result<Tensor> {
        let out = self
            .linear_0
            .forward(label_embeddings)
            .map_err(|e| Error::Parse(format!("label projection 0: {}", e)))?;
        let out = out
            .relu()
            .map_err(|e| Error::Parse(format!("label relu: {}", e)))?;
        self.linear_3
            .forward(&out)
            .map_err(|e| Error::Parse(format!("label projection 3: {}", e)))
    }
}

// =============================================================================
// Span-Label Matcher
// =============================================================================

/// Computes similarity between spans and labels.
#[cfg(feature = "candle")]
pub struct SpanLabelMatcher {
    temperature: f64,
}

#[cfg(feature = "candle")]
impl SpanLabelMatcher {
    /// Create a new span-label matcher with temperature scaling.
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Match spans to labels via cosine similarity.
    ///
    /// # Arguments
    /// * `span_embeddings` - [batch, num_spans, hidden]
    /// * `label_embeddings` - [num_labels, hidden]
    ///
    /// # Returns
    /// [batch, num_spans, num_labels] scores in [0, 1]
    pub fn forward(&self, span_embeddings: &Tensor, label_embeddings: &Tensor) -> Result<Tensor> {
        let span_norm = l2_normalize(span_embeddings, D::Minus1)?;
        let label_norm = l2_normalize(label_embeddings, D::Minus1)?;

        let batch_size = span_norm.dims()[0];
        let label_t = label_norm.t()?;
        let label_t = label_t.unsqueeze(0)?.broadcast_as((
            batch_size,
            label_t.dims()[0],
            label_t.dims()[1],
        ))?;

        let scores = span_norm.matmul(&label_t)?;
        let scaled = (scores * self.temperature)?;

        candle_nn::ops::sigmoid(&scaled).map_err(|e| Error::Parse(format!("sigmoid: {}", e)))
    }
}

#[cfg(feature = "candle")]
pub(crate) fn l2_normalize(tensor: &Tensor, dim: D) -> Result<Tensor> {
    let norm = tensor.sqr()?.sum(dim)?.sqrt()?;
    let norm = norm.unsqueeze(D::Minus1)?;
    // Clamp norm to prevent division by zero
    let norm_clamped = norm
        .clamp(1e-12, f32::MAX)
        .map_err(|e| Error::Parse(format!("clamp: {}", e)))?;
    tensor
        .broadcast_div(&norm_clamped)
        .map_err(|e| Error::Parse(format!("l2_normalize: {}", e)))
}
