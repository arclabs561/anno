// SPDX-License-Identifier: MIT OR Apache-2.0
use anno::core::grounded::html_source::source_sha256;
use serde::{Deserialize, Serialize};
use std::{fs, io::ErrorKind, path::Path};

#[derive(Serialize, Deserialize)]
pub(super) struct BratPair {
    version: u8,
    annotation_sha256: String,
    text_sha256: String,
}

impl BratPair {
    pub fn new(annotation: &str, text: &str) -> Self {
        Self {
            version: 1,
            annotation_sha256: source_sha256(annotation),
            text_sha256: source_sha256(text),
        }
    }

    pub fn read(annotation: &Path) -> Result<Option<Self>, String> {
        let path = annotation.with_extension("anno-pair");
        match fs::symlink_metadata(&path) {
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("Cannot inspect {}: {e}", path.display())),
            Ok(_) => {}
        }
        let bytes = fs::read(&path).map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
        let pair: Self = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Invalid {}: {e}", path.display()))?;
        if pair.version != 1 {
            return Err(format!(
                "Unsupported manifest version {} in {}",
                pair.version,
                path.display()
            ));
        }
        Ok(Some(pair))
    }

    pub fn verify(&self, annotation: &str, text: &str) -> Result<(), String> {
        if source_sha256(annotation) != self.annotation_sha256 {
            return Err("annotation digest differs from the BRAT pair manifest".into());
        }
        if source_sha256(text) != self.text_sha256 {
            return Err("source text digest differs from the BRAT pair manifest".into());
        }
        Ok(())
    }
}
