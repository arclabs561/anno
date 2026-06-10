//! GLiNER2 inference heads (Candle).
//!
//! M5b wires up the four parametric heads needed for the forward pass:
//! [`span_rep::SpanRep`], [`count_lstm::CountLstmFixed`],
//! [`count_pred::CountPred`], and [`classifier::Classifier`]. The other
//! three modules in this directory ([`token_gather`], [`schema_gather`],
//! [`scorer`]) are parameter-free utilities used by the pipeline.

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub mod classifier;
pub mod count_lstm;
pub mod count_pred;
pub mod schema_gather;
pub mod scorer;
pub mod span_rep;
pub mod token_gather;

/// Container for the four parametric inference heads.
pub struct AllHeads {
    pub span_rep: span_rep::SpanRep,
    pub count_lstm: count_lstm::CountLstmFixed,
    pub count_pred: count_pred::CountPred,
    pub classifier: classifier::Classifier,
}

impl AllHeads {
    /// Load all four heads' weights from a single safetensors file.
    ///
    /// Expects the standard `fastino/gliner2-multi-v1` key layout:
    ///   - `span_rep.span_rep_layer.*`  → [`span_rep::SpanRep`]
    ///   - `count_embed.*`              → [`count_lstm::CountLstmFixed`]
    ///   - `count_pred.*`               → [`count_pred::CountPred`]
    ///   - `classifier.*`               → [`classifier::Classifier`]
    ///
    /// See `docs/dev-notes/gliner2-multi-v1-safetensors-keys.md` for the
    /// authoritative key reference.
    pub fn from_safetensors(weights_path: &Path, device: &Device) -> crate::Result<Self> {
        // SAFETY: VarBuilder::from_mmaped_safetensors mmap-reads the
        // weights file. Safe as long as the file isn't mutated under us —
        // Candle's standard pattern (matches `encoder::Encoder`).
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device) }
                .map_err(|e| {
                    crate::Error::Backend(format!("gliner2_fastino_candle: heads safetensors: {e}"))
                })?;

        let span_rep = span_rep::SpanRep::from_var_builder(&vb.pp("span_rep").pp("span_rep_layer"))
            .map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: span_rep load: {e}"))
            })?;

        let count_lstm =
            count_lstm::CountLstmFixed::from_var_builder(&vb.pp("count_embed"), device).map_err(
                |e| crate::Error::Backend(format!("gliner2_fastino_candle: count_embed load: {e}")),
            )?;

        let count_pred =
            count_pred::CountPred::from_var_builder(&vb.pp("count_pred")).map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: count_pred load: {e}"))
            })?;

        let classifier =
            classifier::Classifier::from_var_builder(&vb.pp("classifier")).map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: classifier load: {e}"))
            })?;

        Ok(Self {
            span_rep,
            count_lstm,
            count_pred,
            classifier,
        })
    }

    /// Load all four heads from an already-built [`VarBuilder`].
    ///
    /// Used by [`super::GLiNER2FastinoCandle::load_adapter`] after the
    /// LoRA merge has produced a `HashMap<String, Tensor>` wrapped into
    /// a `VarBuilder::from_tensors`. The VarBuilder must be rooted at
    /// the model's top level (so that `vb.pp("span_rep").pp("span_rep_layer")`
    /// resolves correctly).
    pub fn from_var_builder(vb: VarBuilder<'_>, device: &Device) -> crate::Result<Self> {
        let span_rep = span_rep::SpanRep::from_var_builder(&vb.pp("span_rep").pp("span_rep_layer"))
            .map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: span_rep load (vb): {e}"))
            })?;

        let count_lstm =
            count_lstm::CountLstmFixed::from_var_builder(&vb.pp("count_embed"), device).map_err(
                |e| {
                    crate::Error::Backend(format!(
                        "gliner2_fastino_candle: count_embed load (vb): {e}"
                    ))
                },
            )?;

        let count_pred =
            count_pred::CountPred::from_var_builder(&vb.pp("count_pred")).map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: count_pred load (vb): {e}"))
            })?;

        let classifier =
            classifier::Classifier::from_var_builder(&vb.pp("classifier")).map_err(|e| {
                crate::Error::Backend(format!("gliner2_fastino_candle: classifier load (vb): {e}"))
            })?;

        Ok(Self {
            span_rep,
            count_lstm,
            count_pred,
            classifier,
        })
    }
}
