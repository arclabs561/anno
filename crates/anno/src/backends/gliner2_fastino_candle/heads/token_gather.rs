//! `token_gather` — non-parametric utility head.
//!
//! Performs `index_select` on encoder hidden states at word-start
//! positions. See `docs/dev-notes/gliner2-multi-v1-forward-pass.md`
//! §2 step 1 for the spec.

use candle_core::{Result, Tensor};

/// Stateless token-gather head. Holds no parameters.
pub struct TokenGather;

impl TokenGather {
    /// `hidden_states[0, word_indices, :]` → `[1, num_words, H]`.
    ///
    /// * `hidden_states`: `[1, S, H]` — encoder output.
    /// * `word_indices`: `[num_words]` (i64 / u32) — token-level
    ///   start positions for each word.
    pub fn forward(&self, hidden_states: &Tensor, word_indices: &Tensor) -> Result<Tensor> {
        // Squeeze batch dim, gather, restore batch dim.
        let h = hidden_states.squeeze(0)?; // [S, H]
        let gathered = h.index_select(word_indices, 0)?; // [num_words, H]
        gathered.unsqueeze(0) // [1, num_words, H]
    }
}
