//! Witness type for confidence values bounded to [0.0, 1.0].
//!
//! # What Confidence Actually Means
//!
//! Different NER backends compute confidence in fundamentally different ways.
//! These numbers are NOT directly comparable!
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                    CONFIDENCE ACROSS BACKENDS                            │
//! ├──────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  PATTERN NER: "Did the regex match?"                                     │
//! │  ────────────────────────────────────                                    │
//! │                                                                          │
//! │  • 0.95-0.99 = Regex matched (deterministic)                             │
//! │  • Confidence reflects pattern complexity, NOT uncertainty               │
//! │                                                                          │
//! │    Email pattern matched? → 0.98                                         │
//! │    Date pattern matched?  → 0.95                                         │
//! │                                                                          │
//! │  This is CERTAINTY, not probability.                                     │
//! │  If the pattern fires, it's almost always correct.                       │
//! │                                                                          │
//! │  ──────────────────────────────────────────────────────────────────────  │
//! │                                                                          │
//! │  STATISTICAL NER: "How many heuristics agreed?"                          │
//! │  ───────────────────────────────────────────────                         │
//! │                                                                          │
//! │  • Score = (capitalization + context + gazetteer) / weights              │
//! │  • Range: typically 0.4 - 0.8                                            │
//! │                                                                          │
//! │    "Dr. Smith" → 0.72 (title + capitalization)                           │
//! │    "Apple"     → 0.55 (capitalization only, ambiguous)                   │
//! │                                                                          │
//! │  This is a HEURISTIC BLEND.                                              │
//! │  Higher = more features matched, but not a probability.                  │
//! │                                                                          │
//! │  ──────────────────────────────────────────────────────────────────────  │
//! │                                                                          │
//! │  NEURAL NER (BERT/GLiNER): "Softmax probability"                         │
//! │  ────────────────────────────────────────────────                        │
//! │                                                                          │
//! │  • softmax([logit_PER, logit_ORG, logit_LOC, ...])                       │
//! │  • Range: 0.0 - 1.0, calibrated to approximate probability               │
//! │                                                                          │
//! │    "John"  → PER: 0.94, ORG: 0.03, LOC: 0.03                             │
//! │    "Apple" → ORG: 0.52, PER: 0.01, LOC: 0.47  (ambiguous!)               │
//! │                                                                          │
//! │  This is a CALIBRATED probability (ideally).                             │
//! │  Models with temperature scaling are better calibrated.                  │
//! │                                                                          │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # The Comparison Problem
//!
//! ```text
//! NEVER DO THIS:
//!
//!   PatternNER says EMAIL with 0.98 confidence
//!   HeuristicNER says ORG with 0.55 confidence
//!
//!   "0.98 > 0.55, so EMAIL is more likely!"  ← WRONG!
//!
//! These scales are incompatible:
//!
//!   • PatternNER's 0.98 means "regex matched, nearly certain"
//!   • HeuristicNER's 0.55 means "some features matched, unsure"
//!
//! Comparing them is like comparing °C to °F to Kelvin.
//! Same name (confidence), different scales.
//!
//! ────────────────────────────────────────────────────────────────────────────
//!
//! WHAT TO DO INSTEAD:
//!
//! 1. Use conflict resolution strategies (Priority, LongestSpan)
//! 2. Calibrate scores if mixing backends
//! 3. Threshold per-backend: Pattern > 0.9, Neural > 0.5
//! ```
//!
//! # When to Trust Confidence
//!
//! ```text
//! ┌───────────────┬─────────────────────────────────────────────────────────┐
//! │ Backend       │ When confidence is reliable                            │
//! ├───────────────┼─────────────────────────────────────────────────────────┤
//! │ PatternNER    │ Always (deterministic). 0.95+ means pattern matched.   │
//! │ HeuristicNER│ Use as ranking within backend, not absolute truth.     │
//! │ BERT-NER      │ Reasonably calibrated for in-domain data.              │
//! │ GLiNER        │ Good for ranking, less calibrated for absolute probs.  │
//! └───────────────┴─────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// A confidence score guaranteed to be in the range [0.0, 1.0].
///
/// This is a "witness type" - its existence proves the value is valid.
/// Once you have a `Confidence`, you never need to check bounds again.
///
/// # Construction
///
/// - [`Confidence::new`]: Returns `None` if out of range (strict parsing)
/// - [`Confidence::saturating`]: Clamps to [0, 1] (lenient, never fails)
/// - [`Confidence::try_from`]: Returns `Err` if out of range
///
/// # Zero-Cost Abstraction
///
/// `Confidence` is `#[repr(transparent)]`, meaning it has the exact same
/// memory layout as `f64`. There is no runtime overhead.
///
/// # Example
///
/// ```rust
/// use anno::types::Confidence;
///
/// // Strict: fail on invalid input
/// assert!(Confidence::new(0.5).is_some());
/// assert!(Confidence::new(1.5).is_none());
///
/// // Lenient: clamp to valid range
/// let conf = Confidence::saturating(1.5);
/// assert_eq!(conf.get(), 1.0);
///
/// // Use with Entity - convert to f64 with .get()
/// use anno::{Entity, EntityType};
/// let entity = Entity::new("test", EntityType::Person, 0, 4, conf.get());
/// ```
#[derive(Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Confidence(f64);

impl Confidence {
    /// The minimum valid confidence value.
    pub const MIN: Self = Self(0.0);

    /// The maximum valid confidence value.
    pub const MAX: Self = Self(1.0);

    /// A "perfect" confidence of 1.0 (deterministic/pattern-based extraction).
    pub const CERTAIN: Self = Self(1.0);

    /// A "no information" confidence of 0.5 (maximum entropy).
    pub const UNCERTAIN: Self = Self(0.5);

    /// Create a confidence score, returning `None` if out of range.
    ///
    /// Use this when invalid values should be handled explicitly.
    #[must_use]
    #[inline]
    pub fn new(value: f64) -> Option<Self> {
        if (0.0..=1.0).contains(&value) && !value.is_nan() {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Create a confidence score, clamping to [0.0, 1.0].
    ///
    /// Use this when you want lenient handling of out-of-range values.
    /// NaN is treated as 0.0.
    #[must_use]
    #[inline]
    pub fn saturating(value: f64) -> Self {
        if value.is_nan() {
            Self(0.0)
        } else {
            Self(value.clamp(0.0, 1.0))
        }
    }

    /// Create a confidence score from a percentage (0-100).
    #[must_use]
    #[inline]
    pub fn from_percent(percent: f64) -> Option<Self> {
        Self::new(percent / 100.0)
    }

    /// Get the inner value (guaranteed to be in [0.0, 1.0]).
    #[must_use]
    #[inline]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Convert to percentage (0-100).
    #[must_use]
    #[inline]
    pub fn as_percent(self) -> f64 {
        self.0 * 100.0
    }

    /// Check if this is "high confidence" (>= 0.9).
    #[must_use]
    #[inline]
    pub fn is_high(self) -> bool {
        self.0 >= 0.9
    }

    /// Check if this is "low confidence" (< 0.5).
    #[must_use]
    #[inline]
    pub fn is_low(self) -> bool {
        self.0 < 0.5
    }

    /// Linear interpolation between two confidence values.
    ///
    /// `t = 0.0` returns `self`, `t = 1.0` returns `other`.
    #[must_use]
    #[inline]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self::saturating(self.0 * (1.0 - t) + other.0 * t)
    }

    /// Combine two confidence scores (geometric mean).
    ///
    /// Geometric mean penalizes low scores more than arithmetic mean,
    /// which is appropriate for independent confidence estimates.
    #[must_use]
    #[inline]
    pub fn combine(self, other: Self) -> Self {
        Self((self.0 * other.0).sqrt())
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self::CERTAIN
    }
}

impl fmt::Debug for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Confidence({:.4})", self.0)
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}%", self.0 * 100.0)
    }
}

/// Error when trying to create a Confidence from an invalid value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceError {
    /// The invalid value that was provided.
    pub value: f64,
}

impl fmt::Display for ConfidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "confidence value {} is outside valid range [0.0, 1.0]",
            self.value
        )
    }
}

impl std::error::Error for ConfidenceError {}

impl TryFrom<f64> for Confidence {
    type Error = ConfidenceError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value).ok_or(ConfidenceError { value })
    }
}

impl From<Confidence> for f64 {
    #[inline]
    fn from(conf: Confidence) -> Self {
        conf.0
    }
}

impl PartialEq<f64> for Confidence {
    fn eq(&self, other: &f64) -> bool {
        (self.0 - other).abs() < f64::EPSILON
    }
}

impl PartialOrd<f64> for Confidence {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

/// Type alias for `Confidence` when used in probabilistic contexts.
pub type Probability = Confidence;

/// Type alias for generic unit interval values.
pub type UnitInterval = Confidence;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        assert!(Confidence::new(0.0).is_some());
        assert!(Confidence::new(0.5).is_some());
        assert!(Confidence::new(1.0).is_some());
    }

    #[test]
    fn new_invalid() {
        assert!(Confidence::new(-0.1).is_none());
        assert!(Confidence::new(1.1).is_none());
        assert!(Confidence::new(f64::NAN).is_none());
        assert!(Confidence::new(f64::INFINITY).is_none());
    }

    #[test]
    fn saturating_clamps() {
        assert_eq!(Confidence::saturating(0.5).get(), 0.5);
        assert_eq!(Confidence::saturating(-1.0).get(), 0.0);
        assert_eq!(Confidence::saturating(2.0).get(), 1.0);
        assert_eq!(Confidence::saturating(f64::NAN).get(), 0.0);
    }

    #[test]
    fn from_percent_works() {
        let conf = Confidence::from_percent(85.0).unwrap();
        assert!((conf.get() - 0.85).abs() < 1e-10);
        assert!(Confidence::from_percent(150.0).is_none());
    }

    #[test]
    fn predicates() {
        assert!(Confidence::new(0.95).unwrap().is_high());
        assert!(!Confidence::new(0.85).unwrap().is_high());
        assert!(Confidence::new(0.3).unwrap().is_low());
        assert!(!Confidence::new(0.6).unwrap().is_low());
    }

    #[test]
    fn lerp_bounded() {
        let a = Confidence::new(0.0).unwrap();
        let b = Confidence::new(1.0).unwrap();
        assert!((a.lerp(b, 0.0).get() - 0.0).abs() < 1e-10);
        assert!((a.lerp(b, 0.5).get() - 0.5).abs() < 1e-10);
        assert!((a.lerp(b, 1.0).get() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn combine_geometric_mean() {
        let a = Confidence::new(0.8).unwrap();
        let b = Confidence::new(0.8).unwrap();
        assert!((a.combine(b).get() - 0.8).abs() < 1e-10);

        let c = Confidence::new(1.0).unwrap();
        let d = Confidence::new(0.0).unwrap();
        assert!((c.combine(d).get() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn try_from_f64() {
        let ok: Result<Confidence, _> = 0.5_f64.try_into();
        assert!(ok.is_ok());

        let err: Result<Confidence, _> = 1.5_f64.try_into();
        assert!(err.is_err());
    }

    #[test]
    fn display_format() {
        let conf = Confidence::new(0.856).unwrap();
        assert_eq!(format!("{}", conf), "85.6%");
    }

    #[test]
    fn serde_roundtrip() {
        let conf = Confidence::new(0.85).unwrap();
        let json = serde_json::to_string(&conf).unwrap();
        assert_eq!(json, "0.85");
        let restored: Confidence = serde_json::from_str(&json).unwrap();
        assert!((restored.get() - 0.85).abs() < 1e-10);
    }

    #[test]
    fn constants() {
        assert_eq!(Confidence::MIN.get(), 0.0);
        assert_eq!(Confidence::MAX.get(), 1.0);
        assert_eq!(Confidence::CERTAIN.get(), 1.0);
        assert_eq!(Confidence::UNCERTAIN.get(), 0.5);
    }
}
