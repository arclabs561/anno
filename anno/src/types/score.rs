//! Witness type for f32 neural network output scores bounded to [0.0, 1.0].

use super::Confidence;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A score guaranteed to be in the range [0.0, 1.0] (f32 precision).
///
/// This is the f32 equivalent of `Confidence`, designed for neural network
/// outputs where f32 is standard (tensors, embeddings, similarity scores).
///
/// # Why f32?
///
/// Neural networks typically operate in f32:
/// - GPU hardware optimized for f32/f16
/// - Embeddings use f32 (768 floats per token)
/// - Logits/softmax outputs are f32
///
/// # Construction
///
/// - [`Score::new`]: Returns `None` if out of range
/// - [`Score::saturating`]: Clamps to [0, 1]
/// - [`Score::from_logit`]: Applies sigmoid (common for NER)
///
/// # Zero-Cost Abstraction
///
/// `Score` is `#[repr(transparent)]`, meaning it has the exact same
/// memory layout as `f32`. There is no runtime overhead.
///
/// # Example
///
/// ```rust
/// use anno::types::Score;
///
/// // From neural network output
/// let logit = 2.5f32;
/// let score = Score::from_logit(logit);
/// assert!(score.get() > 0.9);
///
/// // Convert to Confidence (f64) for API consistency
/// let conf = score.to_confidence();
/// ```
#[derive(Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Score(f32);

impl Score {
    /// The minimum valid score.
    pub const MIN: Self = Self(0.0);

    /// The maximum valid score.
    pub const MAX: Self = Self(1.0);

    /// Create a score, returning `None` if out of range.
    #[must_use]
    #[inline]
    pub fn new(value: f32) -> Option<Self> {
        if (0.0..=1.0).contains(&value) && !value.is_nan() {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Create a score, clamping to [0.0, 1.0].
    #[must_use]
    #[inline]
    pub fn saturating(value: f32) -> Self {
        if value.is_nan() {
            Self(0.0)
        } else {
            Self(value.clamp(0.0, 1.0))
        }
    }

    /// Create from a raw logit by applying sigmoid.
    ///
    /// sigmoid(x) = 1 / (1 + e^(-x))
    #[must_use]
    #[inline]
    pub fn from_logit(logit: f32) -> Self {
        Self(1.0 / (1.0 + (-logit).exp()))
    }

    /// Create from a temperature-scaled logit.
    ///
    /// Higher temperature (> 1) = softer distribution.
    /// Lower temperature (< 1) = sharper distribution.
    #[must_use]
    #[inline]
    pub fn from_logit_with_temperature(logit: f32, temperature: f32) -> Self {
        let scaled = if temperature > 0.0 {
            logit / temperature
        } else {
            logit
        };
        Self::from_logit(scaled)
    }

    /// Get the inner value (guaranteed to be in [0.0, 1.0]).
    #[must_use]
    #[inline]
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Convert to f64 `Confidence`.
    #[must_use]
    #[inline]
    pub fn to_confidence(self) -> Confidence {
        Confidence::saturating(self.0 as f64)
    }

    /// Check if this is "high confidence" (>= 0.9).
    #[must_use]
    #[inline]
    pub fn is_high(self) -> bool {
        self.0 >= 0.9
    }

    /// Check if this passes a threshold.
    #[must_use]
    #[inline]
    pub fn passes(self, threshold: f32) -> bool {
        self.0 >= threshold
    }
}

impl Default for Score {
    fn default() -> Self {
        Self::MAX
    }
}

impl fmt::Debug for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Score({:.4})", self.0)
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}%", self.0 * 100.0)
    }
}

impl From<Score> for f32 {
    #[inline]
    fn from(score: Score) -> Self {
        score.0
    }
}

impl From<Score> for Confidence {
    #[inline]
    fn from(score: Score) -> Self {
        score.to_confidence()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        assert!(Score::new(0.0).is_some());
        assert!(Score::new(0.5).is_some());
        assert!(Score::new(1.0).is_some());
    }

    #[test]
    fn new_invalid() {
        assert!(Score::new(-0.1).is_none());
        assert!(Score::new(1.1).is_none());
        assert!(Score::new(f32::NAN).is_none());
    }

    #[test]
    fn saturating_clamps() {
        assert_eq!(Score::saturating(0.5).get(), 0.5);
        assert_eq!(Score::saturating(-1.0).get(), 0.0);
        assert_eq!(Score::saturating(2.0).get(), 1.0);
        assert_eq!(Score::saturating(f32::NAN).get(), 0.0);
    }

    #[test]
    fn from_logit_sigmoid() {
        // sigmoid(0) = 0.5
        let score = Score::from_logit(0.0);
        assert!((score.get() - 0.5).abs() < 0.001);

        // sigmoid(large) close to 1.0
        let high = Score::from_logit(10.0);
        assert!(high.get() > 0.99);

        // sigmoid(negative) close to 0.0
        let low = Score::from_logit(-10.0);
        assert!(low.get() < 0.01);
    }

    #[test]
    fn from_logit_temperature() {
        let soft = Score::from_logit_with_temperature(2.0, 5.0);
        let sharp = Score::from_logit_with_temperature(2.0, 0.1);
        // Sharp should be closer to 1.0
        assert!(sharp.get() > soft.get());
    }

    #[test]
    fn to_confidence_preserves_value() {
        let score = Score::new(0.85).unwrap();
        let conf = score.to_confidence();
        assert!((conf.get() - 0.85).abs() < 0.001);
    }

    #[test]
    fn predicates() {
        assert!(Score::new(0.95).unwrap().is_high());
        assert!(!Score::new(0.85).unwrap().is_high());
        assert!(Score::new(0.7).unwrap().passes(0.5));
        assert!(!Score::new(0.4).unwrap().passes(0.5));
    }

    #[test]
    fn display_format() {
        let score = Score::new(0.856).unwrap();
        assert_eq!(format!("{}", score), "85.6%");
    }
}
