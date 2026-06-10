//! Backend-local error type for `gliner2_fastino`. Mapped into `anno::Error`
//! at the public API boundary.

use std::path::PathBuf;
use thiserror::Error;

/// Backend-local error type for fastino GLiNER2.
#[derive(Debug, Error)]
pub enum Error {
    /// Tokenizer file not found.
    #[error("tokenizer.json not found at {0}")]
    TokenizerMissing(PathBuf),

    /// Required config field missing.
    #[error("config.json missing required field `{field}` for fastino GLiNER2 model")]
    ConfigFieldMissing {
        /// The missing field name.
        field: &'static str,
    },

    /// Required special token missing.
    #[error(
        "missing required special token `{token}` in tokenizer.json — \
         fastino GLiNER2 models require [P]/[E]/[C]/[L]/[R]/[SEP_STRUCT]/[SEP_TEXT]"
    )]
    SpecialTokenMissing {
        /// The missing token.
        token: &'static str,
    },

    /// LoRA adapter not supported (Phase 1).
    #[error(
        "directory at {path} contains a LoRA adapter (adapter_config.json). \
         runtime adapter hot-swap is not supported in Phase 1. \
         merge the adapter into the base model and re-export to ONNX with: \
         `python scripts/gliner2_export_onnx.py --base BASE --lora-adapter {path:?} --output OUTPUT.onnx`. \
         See issue #18 / Phase 4 for runtime hot-swap status."
    )]
    LoraAdapterNotSupported {
        /// Path to the adapter directory.
        path: PathBuf,
    },

    /// ONNX Runtime session error.
    #[error("ort session error: {0}")]
    Ort(#[from] ort::Error),

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Tokenizer processing error.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Config JSON parsing error.
    #[error("config parse error: {0}")]
    ConfigParse(#[from] serde_json::Error),
}

impl From<Error> for crate::Error {
    fn from(e: Error) -> Self {
        crate::Error::Backend(format!("gliner2_fastino: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_error_message_contains_script_path_and_phase4_pointer() {
        let e = Error::LoraAdapterNotSupported {
            path: PathBuf::from("/tmp/my_adapter"),
        };
        let msg = e.to_string();
        assert!(
            msg.contains("scripts/gliner2_export_onnx.py"),
            "missing script path: {msg}"
        );
        assert!(msg.contains("--lora-adapter"), "missing flag in msg: {msg}");
        assert!(
            msg.contains("Phase 4") || msg.contains("hot-swap"),
            "missing future-state pointer: {msg}"
        );
    }
}
