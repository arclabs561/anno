//! Cache management for CLI.
//!
//! Provides transparent caching of extraction results with smart invalidation.
//!
//! # Architecture
//!
//! The caching system has two layers:
//! - **Result cache**: Caches extraction results by input hash
//! - **Model cache**: Caches downloaded models and datasets
//!
//! # Usage
//!
//! ```bash
//! # List cache contents
//! anno cache list
//!
//! # Clear all cached results
//! anno cache clear
//!
//! # Show cache statistics
//! anno cache stats
//! ```
//!
//! See [`crate::cli::commands::cache`] for the command implementation.

// (module intentionally empty; commands live under `cli::commands::cache`)
