//! # anno
//!
//! Published facade crate for the `anno` workspace.
//!
//! This package provides:
//! - the `anno` library API (re-exported from the internal `anno-lib` crate)
//! - the `anno` CLI (see `crates/anno-cli/`)
//!
//! Internal crates remain workspace-private (not separately published) for now.

#![warn(missing_docs)]

pub use anno_lib::*;
