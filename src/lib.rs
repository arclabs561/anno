//! # anno
//!
//! Information extraction for unstructured text: named entity recognition (NER),
//! coreference resolution, relation extraction, PII detection, and structured
//! pattern extraction.
//!
//! This is the published facade crate for the `anno` workspace.  It re-exports
//! the internal `anno-lib` library API.  The full CLI lives in `crates/anno-cli/`.
//!
//! - **NER**: variable-length spans with character offsets (Unicode scalar values).
//! - **Coreference**: mention clusters ("tracks") within a single document.
//! - **Patterns**: dates, monetary amounts, emails, URLs, phone numbers.
//!
//! Internal crates (`anno-lib`, `anno-core`, `anno-metrics`, `anno-eval`,
//! `anno-cli`, `anno-graph`) are workspace-private and not separately published.

#![warn(missing_docs)]

pub use anno_lib::*;
