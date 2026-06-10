//! Multi-session ONNX container for `gliner2_fastino`. Phase 3.
#![allow(missing_docs)] // implementation internals; public API is on GLiNER2Fastino in mod.rs
//!
//! Adapted from SemplificaAI/gliner2-rs (Apache-2.0):
//! https://github.com/SemplificaAI/gliner2-rs/blob/main/rust_component/src/lib_v2.rs
//! Original: Copyright 2026 Dario Finardi, Semplifica s.r.l.

use crate::backends::gliner2_fastino::errors::Error;
use crate::backends::hf_loader;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Eight ONNX sessions making up the GLiNER2 v2 inference pipeline.
///
/// Each `SessionSlot` is wrapped in `Arc<Mutex<>>` so the engine can hand
/// out `&self` while the closure-style `with_session` API mutates the
/// underlying ort `Session::run`. This mirrors Phase 1's single-Session
/// pattern, applied per role.
pub struct Sessions {
    pub encoder: SessionSlot,
    pub token_gather: SessionSlot,
    pub span_rep: SessionSlot,
    pub schema_gather: SessionSlot,
    pub count_pred_argmax: SessionSlot,
    pub count_lstm_fixed: SessionSlot,
    pub scorer: SessionSlot,
    pub classifier: SessionSlot,
}

/// Single session wrapped in Arc<Mutex<>> with a `with_session` closure
/// API. Identical to Phase 1's `session::Session` but extracted as a
/// reusable type.
#[derive(Debug)]
pub struct SessionSlot {
    inner: Arc<Mutex<ort::session::Session>>,
}

impl SessionSlot {
    pub fn from_path_with_cfg(
        model_path: &Path,
        cfg: hf_loader::OnnxSessionConfig,
    ) -> Result<Self, Error> {
        let session = hf_loader::create_onnx_session(model_path, cfg)
            .map_err(|e| Error::Tokenizer(format!("session {}: {e}", model_path.display())))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(session)),
        })
    }

    #[allow(dead_code)] // convenience ctor; pipeline uses from_path_with_cfg
    pub fn from_path(model_path: &Path) -> Result<Self, Error> {
        Self::from_path_with_cfg(model_path, hf_loader::OnnxSessionConfig::default())
    }

    pub fn with_session<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut ort::session::Session) -> R,
    {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }
}

impl std::fmt::Debug for Sessions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sessions").field("count", &8usize).finish()
    }
}

impl Sessions {
    /// Load all 8 sessions from a directory with custom ONNX session configuration.
    /// Tries the v2 8-graph layouts (`fp32_v2/`, `fp16_v2/`) first, then v1 5-graph
    /// fallbacks (which will fail the all_present check below — kept for future-proofing).
    ///
    /// Phase 3 standard mode does NOT use the `_iobinding.onnx` variants
    /// — those are reserved for Phase 3.5 IOBinding mode.
    #[allow(dead_code)] // convenience ctor; pipeline uses from_dir_with_cfg_mode
    pub fn from_dir_with_cfg(
        model_dir: &Path,
        cfg: hf_loader::OnnxSessionConfig,
    ) -> Result<(Self, std::path::PathBuf), Error> {
        Self::from_dir_with_cfg_mode(model_dir, cfg, super::ExecutionMode::Standard)
    }

    /// Phase 3.5: load with explicit execution mode. When `mode` is
    /// [`super::ExecutionMode::IoBinding`], prefers the `*_iobinding{suffix}`
    /// variant (e.g. `encoder_iobinding_fp32.onnx`) per session and falls
    /// back to the standard variant if the iobinding-specific export is
    /// not shipped. IoBinding is purely a runtime API — both variants are
    /// functionally usable; the suffix exists for upstream exports that
    /// freeze dynamic shapes for better device-side allocation.
    pub fn from_dir_with_cfg_mode(
        model_dir: &Path,
        cfg: hf_loader::OnnxSessionConfig,
        mode: super::ExecutionMode,
    ) -> Result<(Self, std::path::PathBuf), Error> {
        // The 8-session v2 layout lives in `_v2` subdirs only — `fp32/`
        // and `fp16/` are the legacy v1 layout (5 graphs, missing
        // token_gather/schema_gather/count_pred_argmax/count_lstm_fixed).
        // The `all_present` guard below filters incompatible layouts.
        for (subdir, suffix) in [
            ("fp32_v2", "_fp32.onnx"),
            ("fp16_v2", "_fp16.onnx"),
            ("fp32", "_fp32.onnx"), // v1 fallback — likely won't match
            ("fp16", "_fp16.onnx"), // v1 fallback
        ] {
            let try_dir = model_dir.join(subdir);
            if !try_dir.is_dir() {
                continue;
            }
            // Resolve a session's filename. In IoBinding mode, prefer the
            // `_iobinding`-suffixed variant if it exists; fall back to the
            // standard filename otherwise. In Standard mode, always pick
            // the standard filename.
            let resolve = |name: &str| -> std::path::PathBuf {
                if matches!(mode, super::ExecutionMode::IoBinding) {
                    let io = try_dir.join(format!("{name}_iobinding{suffix}"));
                    if io.exists() {
                        return io;
                    }
                }
                try_dir.join(format!("{name}{suffix}"))
            };
            let all_present = [
                "encoder",
                "token_gather",
                "span_rep",
                "schema_gather",
                "count_pred_argmax",
                "count_lstm_fixed",
                "scorer",
                "classifier",
            ]
            .iter()
            .all(|n| resolve(n).exists());
            if !all_present {
                continue;
            }
            return Ok((
                Self {
                    encoder: SessionSlot::from_path_with_cfg(&resolve("encoder"), cfg.clone())?,
                    token_gather: SessionSlot::from_path_with_cfg(
                        &resolve("token_gather"),
                        cfg.clone(),
                    )?,
                    span_rep: SessionSlot::from_path_with_cfg(&resolve("span_rep"), cfg.clone())?,
                    schema_gather: SessionSlot::from_path_with_cfg(
                        &resolve("schema_gather"),
                        cfg.clone(),
                    )?,
                    count_pred_argmax: SessionSlot::from_path_with_cfg(
                        &resolve("count_pred_argmax"),
                        cfg.clone(),
                    )?,
                    count_lstm_fixed: SessionSlot::from_path_with_cfg(
                        &resolve("count_lstm_fixed"),
                        cfg.clone(),
                    )?,
                    scorer: SessionSlot::from_path_with_cfg(&resolve("scorer"), cfg.clone())?,
                    classifier: SessionSlot::from_path_with_cfg(
                        &resolve("classifier"),
                        cfg.clone(),
                    )?,
                },
                try_dir,
            ));
        }
        Err(Error::Tokenizer(format!(
            "no complete v2 session set found under {} (looked in fp32_v2/, fp16_v2/, fp32/, fp16/)",
            model_dir.display()
        )))
    }

    /// Load all 8 sessions from a directory. Tries the v2 8-graph layouts
    /// (`fp32_v2/`, `fp16_v2/`) first, then v1 5-graph fallbacks (which
    /// will fail the all_present check below — kept for future-proofing).
    ///
    /// Phase 3 standard mode does NOT use the `_iobinding.onnx` variants
    /// — those are reserved for Phase 3.5 IOBinding mode.
    #[allow(dead_code)] // convenience ctor; pipeline uses from_dir_with_cfg_mode
    pub fn from_dir(model_dir: &Path) -> Result<(Self, std::path::PathBuf), Error> {
        Self::from_dir_with_cfg(model_dir, hf_loader::OnnxSessionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn from_dir_fails_clearly_on_empty_dir() {
        let dir = tempdir().unwrap();
        let err = Sessions::from_dir(dir.path()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no complete v2 session set"), "got: {msg}");
        assert!(msg.contains("fp32") || msg.contains("fp16"), "got: {msg}");
    }

    #[test]
    fn from_dir_fails_clearly_on_partial_layout() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("fp32_v2")).unwrap();
        // Only encoder present — should not be a "complete set".
        std::fs::write(dir.path().join("fp32_v2/encoder_fp32.onnx"), b"").unwrap();
        let err = Sessions::from_dir(dir.path()).unwrap_err();
        assert!(err.to_string().contains("no complete v2 session set"));
    }

    #[test]
    fn iobinding_mode_prefers_iobinding_variant_when_present() {
        // Phase 3.5 M3: when ExecutionMode::IoBinding is requested AND the
        // `*_iobinding{suffix}` variants exist, they must be preferred over
        // the standard variants. Verified via error-message divergence:
        // Standard mode can't see iobinding-only files → "no complete v2
        // session set"; IoBinding mode finds them, advances past resolution
        // to ONNX load (which fails on parse with a different error).
        let dir = tempdir().unwrap();
        let v2 = dir.path().join("fp32_v2");
        std::fs::create_dir_all(&v2).unwrap();
        for n in [
            "encoder",
            "token_gather",
            "span_rep",
            "schema_gather",
            "count_pred_argmax",
            "count_lstm_fixed",
            "scorer",
            "classifier",
        ] {
            std::fs::write(v2.join(format!("{n}_iobinding_fp32.onnx")), b"").unwrap();
        }

        let err_std = Sessions::from_dir_with_cfg_mode(
            dir.path(),
            hf_loader::OnnxSessionConfig::default(),
            super::super::ExecutionMode::Standard,
        )
        .unwrap_err();
        assert!(
            err_std.to_string().contains("no complete v2 session set"),
            "Standard mode should not see iobinding-only variants. Got: {err_std}"
        );

        let err_io = Sessions::from_dir_with_cfg_mode(
            dir.path(),
            hf_loader::OnnxSessionConfig::default(),
            super::super::ExecutionMode::IoBinding,
        )
        .unwrap_err();
        let msg = err_io.to_string();
        assert!(
            !msg.contains("no complete v2 session set"),
            "IoBinding mode should resolve iobinding variants and advance past 'no complete' check. Got: {msg}"
        );
    }

    #[test]
    fn iobinding_mode_falls_back_to_standard_when_iobinding_missing() {
        // Phase 3.5 M3: IoBinding mode is non-fatal when the model export
        // ships only standard variants. We fall back to `{name}{suffix}`.
        // Verified via error-message divergence: both modes resolve files
        // (all_present passes), then ONNX load fails on parse — neither
        // hits "no complete v2 session set".
        let dir = tempdir().unwrap();
        let v2 = dir.path().join("fp32_v2");
        std::fs::create_dir_all(&v2).unwrap();
        for n in [
            "encoder",
            "token_gather",
            "span_rep",
            "schema_gather",
            "count_pred_argmax",
            "count_lstm_fixed",
            "scorer",
            "classifier",
        ] {
            std::fs::write(v2.join(format!("{n}_fp32.onnx")), b"").unwrap();
        }

        let err_io = Sessions::from_dir_with_cfg_mode(
            dir.path(),
            hf_loader::OnnxSessionConfig::default(),
            super::super::ExecutionMode::IoBinding,
        )
        .unwrap_err();
        let msg = err_io.to_string();
        assert!(
            !msg.contains("no complete v2 session set"),
            "IoBinding mode should fall back to standard variants. Got: {msg}"
        );
    }
}
