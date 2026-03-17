//! Type-safe confidence score constrained to `[0.0, 1.0]`.

use serde::{de, Serialize};
use std::fmt;

/// A confidence score guaranteed to be in `[0.0, 1.0]`.
///
/// Values outside the range are clamped at construction time. This type is
/// serde-transparent (serializes/deserializes as a plain `f64`) for backward
/// compatibility with existing JSON data.
///
/// # Construction
///
/// ```rust
/// use anno_core::Confidence;
///
/// let c = Confidence::new(0.95);
/// assert_eq!(c.value(), 0.95);
///
/// // Out-of-range values are clamped
/// let c = Confidence::new(1.5);
/// assert_eq!(c.value(), 1.0);
///
/// // From f64
/// let c: Confidence = 0.8.into();
/// assert_eq!(c.value(), 0.8);
/// ```
#[derive(Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Confidence(f64);

impl<'de> de::Deserialize<'de> for Confidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let v = f64::deserialize(deserializer)?;
        Ok(Self::new(v))
    }
}

impl Confidence {
    /// Certainty: 1.0.
    pub const ONE: Self = Self(1.0);

    /// No confidence: 0.0.
    pub const ZERO: Self = Self(0.0);

    /// Create a confidence score, clamping to `[0.0, 1.0]`.
    #[must_use]
    #[inline]
    pub fn new(value: f64) -> Self {
        let v = if value.is_nan() { 0.0 } else { value };
        Self(v.clamp(0.0, 1.0))
    }

    /// Get the raw `f64` value.
    #[must_use]
    #[inline]
    pub const fn value(self) -> f64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// JSON Schema (behind `schema` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "schema")]
impl schemars::JsonSchema for Confidence {
    fn schema_name() -> String {
        "Confidence".to_string()
    }

    fn json_schema(gen: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        // Transparent: appears as f64 with [0.0, 1.0] bounds in JSON Schema.
        let mut schema = gen.subschema_for::<f64>().into_object();
        schema.number().minimum = Some(0.0);
        schema.number().maximum = Some(1.0);
        schema.metadata().description = Some("Confidence score in [0.0, 1.0]".to_string());
        schemars::schema::Schema::Object(schema)
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl From<f64> for Confidence {
    #[inline]
    fn from(v: f64) -> Self {
        Self::new(v)
    }
}

impl From<f32> for Confidence {
    #[inline]
    fn from(v: f32) -> Self {
        Self::new(v as f64)
    }
}

impl From<Confidence> for f64 {
    #[inline]
    fn from(c: Confidence) -> f64 {
        c.0
    }
}

impl From<Confidence> for f32 {
    #[inline]
    fn from(c: Confidence) -> f32 {
        c.0 as f32
    }
}

// ---------------------------------------------------------------------------
// Cross-type comparison (so `confidence > 0.5` works without .value())
// ---------------------------------------------------------------------------

impl PartialEq<f64> for Confidence {
    #[inline]
    fn eq(&self, other: &f64) -> bool {
        self.0 == *other
    }
}

impl PartialOrd<f64> for Confidence {
    #[inline]
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialEq<Confidence> for f64 {
    #[inline]
    fn eq(&self, other: &Confidence) -> bool {
        *self == other.0
    }
}

impl PartialOrd<Confidence> for f64 {
    #[inline]
    fn partial_cmp(&self, other: &Confidence) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Arithmetic (results are NOT clamped -- intermediate math may exceed [0,1])
// ---------------------------------------------------------------------------

impl std::ops::Add for Confidence {
    type Output = f64;
    #[inline]
    fn add(self, rhs: Self) -> f64 {
        self.0 + rhs.0
    }
}

impl std::ops::Sub for Confidence {
    type Output = f64;
    #[inline]
    fn sub(self, rhs: Self) -> f64 {
        self.0 - rhs.0
    }
}

impl std::ops::Sub<f64> for Confidence {
    type Output = f64;
    #[inline]
    fn sub(self, rhs: f64) -> f64 {
        self.0 - rhs
    }
}

impl std::ops::Add<f64> for Confidence {
    type Output = f64;
    #[inline]
    fn add(self, rhs: f64) -> f64 {
        self.0 + rhs
    }
}

impl std::ops::Mul<f64> for Confidence {
    type Output = f64;
    #[inline]
    fn mul(self, rhs: f64) -> f64 {
        self.0 * rhs
    }
}

impl std::ops::Mul<Confidence> for f64 {
    type Output = f64;
    #[inline]
    fn mul(self, rhs: Confidence) -> f64 {
        self * rhs.0
    }
}

impl std::ops::Sub<Confidence> for f64 {
    type Output = f64;
    #[inline]
    fn sub(self, rhs: Confidence) -> f64 {
        self - rhs.0
    }
}

impl std::ops::Add<Confidence> for f64 {
    type Output = f64;
    #[inline]
    fn add(self, rhs: Confidence) -> f64 {
        self + rhs.0
    }
}

impl std::iter::Sum for Confidence {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Confidence::new(iter.map(|c| c.0).sum())
    }
}

impl std::iter::Sum<Confidence> for f64 {
    fn sum<I: Iterator<Item = Confidence>>(iter: I) -> f64 {
        iter.map(|c| c.0).sum()
    }
}

impl std::ops::MulAssign<f64> for Confidence {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.0 = (self.0 * rhs).clamp(0.0, 1.0);
    }
}

impl std::ops::Mul<Confidence> for Confidence {
    type Output = f64;
    #[inline]
    fn mul(self, rhs: Self) -> f64 {
        self.0 * rhs.0
    }
}

impl std::ops::Div<f64> for Confidence {
    type Output = f64;
    #[inline]
    fn div(self, rhs: f64) -> f64 {
        self.0 / rhs
    }
}

impl Confidence {
    /// Natural log of the confidence value. Returns `f64` since ln can be negative.
    #[inline]
    pub fn ln(self) -> f64 {
        self.0.ln()
    }

    /// Total comparison (for sorting). Delegates to `f64::total_cmp`.
    #[inline]
    pub fn total_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }

    /// Absolute value (identity for non-negative, but useful in generic code).
    #[inline]
    pub fn abs(self) -> f64 {
        self.0.abs()
    }

    /// Clamp as f64 (useful when intermediate math may have drifted).
    #[inline]
    pub fn clamped(value: f64) -> Self {
        Self::new(value)
    }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

impl fmt::Debug for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Confidence({:.4})", self.0)
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Respect caller's precision (e.g., `{:.2}` for 2 decimals).
        // Default to 4 decimals when no precision is specified.
        if let Some(precision) = f.precision() {
            write!(f, "{:.*}", precision, self.0)
        } else {
            write!(f, "{:.4}", self.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_to_unit_interval() {
        assert_eq!(Confidence::new(-0.5).value(), 0.0);
        assert_eq!(Confidence::new(0.5).value(), 0.5);
        assert_eq!(Confidence::new(1.5).value(), 1.0);
    }

    #[test]
    fn from_f64() {
        let c: Confidence = 0.9.into();
        assert_eq!(c.value(), 0.9);
    }

    #[test]
    fn into_f64() {
        let c = Confidence::new(0.7);
        let v: f64 = c.into();
        assert_eq!(v, 0.7);
    }

    #[test]
    fn cross_type_comparison() {
        let c = Confidence::new(0.8);
        assert!(c > 0.5);
        assert!(c < 0.9);
        assert!(c == 0.8);
    }

    #[test]
    fn cross_type_comparison_reverse() {
        let c = Confidence::new(0.8);
        // f64 on the left, Confidence on the right
        assert!(0.5 < c);
        assert!(0.9 > c);
        assert!(0.8 == c);
        assert!(0.7 != c);
    }

    #[test]
    fn serde_transparent() {
        let c = Confidence::new(0.42);
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(json, "0.42");
        let roundtrip: Confidence = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, c);
    }

    #[test]
    fn constants() {
        assert_eq!(Confidence::ZERO.value(), 0.0);
        assert_eq!(Confidence::ONE.value(), 1.0);
    }

    #[test]
    fn nan_and_infinity_handling() {
        assert_eq!(Confidence::new(f64::NAN).value(), 0.0);
        assert_eq!(Confidence::new(f64::INFINITY).value(), 1.0);
        assert_eq!(Confidence::new(f64::NEG_INFINITY).value(), 0.0);
    }

    #[test]
    fn deserialize_clamps_out_of_range() {
        let over: Confidence = serde_json::from_str("1.5").unwrap();
        assert_eq!(over.value(), 1.0);

        let under: Confidence = serde_json::from_str("-0.5").unwrap();
        assert_eq!(under.value(), 0.0);
    }

    #[test]
    fn serde_roundtrip() {
        let c = Confidence::new(0.73);
        let json = serde_json::to_string(&c).unwrap();
        let back: Confidence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn display_respects_precision() {
        let c = Confidence::new(0.9512);
        assert_eq!(format!("{:.2}", c), "0.95");
        assert_eq!(format!("{:.4}", c), "0.9512");
        assert_eq!(format!("{:.0}", c), "1");
        // Default (no precision specified) = 4 decimals
        assert_eq!(format!("{}", c), "0.9512");
    }
}
