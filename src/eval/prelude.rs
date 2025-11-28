//! Evaluation prelude - commonly used types for quick imports.
//!
//! # Usage
//!
//! ```rust
//! use anno::eval::prelude::*;
//! ```
//!
//! This provides the minimal set of types needed for most evaluation tasks:
//! - `EvalReport` and `ReportBuilder` for unified evaluation
//! - `TestCase` and `GoldEntity` for test data
//! - Core metrics types
//!
//! For specialized analysis (bias, calibration, etc.), import from specific modules.

// Core evaluation report (recommended entry point)
pub use super::report::{
    EvalReport, ReportBuilder, TestCase, 
    GoldEntity as ReportGoldEntity, 
    CoreMetrics, TypeMetrics as ReportTypeMetrics,
    Priority, Recommendation,
};

// Synthetic data generation
pub use super::synthetic_gen::{Template, generate_test_cases, standard_test_set};

// Basic types
pub use super::GoldEntity;
pub use super::EvalMode;

// Re-export Model trait for convenience
pub use crate::Model;

