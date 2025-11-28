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
//! For specialized analysis, import from specific modules:
//! - `anno::eval::gender_bias` (requires `eval-bias` feature)
//! - `anno::eval::calibration` (requires `eval-advanced` feature)

// =============================================================================
// CORE (always available with `eval` feature)
// =============================================================================

// Unified evaluation report (recommended entry point)
pub use super::report::{
    CoreMetrics, EvalReport, SimpleGoldEntity, Priority, Recommendation,
    ReportBuilder, TestCase, TypeMetrics as ReportTypeMetrics,
};

// Synthetic data generation
pub use super::synthetic_gen::{generate_test_cases, standard_test_set, Template};

// Basic types
pub use super::EvalMode;
pub use super::GoldEntity;
pub use super::MetricValue;
pub use super::TypeMetrics;

// Coreference evaluation
pub use super::coref::{CorefChain, CorefDocument, Mention};
pub use super::coref_resolver::SimpleCorefResolver;

// Standard evaluator
pub use super::evaluator::{AveragingMode, NERAggregateMetrics, NEREvaluator, StandardNEREvaluator};

// Re-export Model trait for convenience
pub use crate::Model;

// =============================================================================
// BIAS ANALYSIS (requires `eval-bias` feature)
// =============================================================================

#[cfg(feature = "eval-bias")]
pub use super::demographic_bias::{DemographicBiasEvaluator, DemographicBiasResults};
#[cfg(feature = "eval-bias")]
pub use super::gender_bias::{GenderBiasEvaluator, GenderBiasResults};
#[cfg(feature = "eval-bias")]
pub use super::length_bias::{EntityLengthEvaluator, LengthBiasResults};
#[cfg(feature = "eval-bias")]
pub use super::temporal_bias::{TemporalBiasEvaluator, TemporalBiasResults};

// =============================================================================
// ADVANCED ANALYSIS (requires `eval-advanced` feature)
// =============================================================================

#[cfg(feature = "eval-advanced")]
pub use super::calibration::{CalibrationEvaluator, CalibrationResults};
#[cfg(feature = "eval-advanced")]
pub use super::error_analysis::{ErrorAnalyzer, ErrorReport};
#[cfg(feature = "eval-advanced")]
pub use super::robustness::{RobustnessEvaluator, RobustnessResults};
#[cfg(feature = "eval-advanced")]
pub use super::threshold_analysis::{ThresholdAnalyzer, ThresholdCurve};

